; ============================================================================
; onnx_runtime.pbi - optional native host-side ONNX Runtime bridge
; ----------------------------------------------------------------------------
; This bridge is used only while compiling, to execute a bounded shape trace
; when tensor extents depend on input values. Generated programs never link to
; ONNX Runtime. The compiler loads the versioned C ABI dynamically, so there is
; no import library and no Python process or package dependency.
;
; Function-table indexes below are pinned to ORT_API_VERSION 29, from the
; official ONNX Runtime 1.29 C header. PmoOrtLoad rejects a runtime that cannot
; supply that exact API table.
; ============================================================================

#PMO_ORT_API_VERSION = 29

Enumeration PmoOrtApiIndex
  #PMO_ORT_CREATE_STATUS = 0
  #PMO_ORT_GET_ERROR_CODE = 1
  #PMO_ORT_GET_ERROR_MESSAGE = 2
  #PMO_ORT_CREATE_ENV = 3
  #PMO_ORT_CREATE_SESSION = 7
  #PMO_ORT_CREATE_SESSION_FROM_ARRAY = 8
  #PMO_ORT_RUN = 9
  #PMO_ORT_CREATE_SESSION_OPTIONS = 10
  #PMO_ORT_SET_GRAPH_OPTIMIZATION = 23
  #PMO_ORT_SESSION_GET_INPUT_COUNT = 30
  #PMO_ORT_SESSION_GET_OUTPUT_COUNT = 31
  #PMO_ORT_SESSION_GET_INPUT_NAME = 36
  #PMO_ORT_SESSION_GET_OUTPUT_NAME = 37
  #PMO_ORT_CREATE_TENSOR_WITH_DATA = 49
  #PMO_ORT_GET_TENSOR_MUTABLE_DATA = 51
  #PMO_ORT_GET_TENSOR_ELEMENT_TYPE = 60
  #PMO_ORT_GET_DIMENSIONS_COUNT = 61
  #PMO_ORT_GET_DIMENSIONS = 62
  #PMO_ORT_GET_TENSOR_ELEMENT_COUNT = 64
  #PMO_ORT_GET_TENSOR_TYPE_AND_SHAPE = 65
  #PMO_ORT_CREATE_CPU_MEMORY_INFO = 69
  #PMO_ORT_ALLOCATOR_FREE = 76
  #PMO_ORT_GET_DEFAULT_ALLOCATOR = 78
  #PMO_ORT_RELEASE_ENV = 92
  #PMO_ORT_RELEASE_STATUS = 93
  #PMO_ORT_RELEASE_MEMORY_INFO = 94
  #PMO_ORT_RELEASE_SESSION = 95
  #PMO_ORT_RELEASE_VALUE = 96
  #PMO_ORT_RELEASE_TENSOR_TYPE_AND_SHAPE = 99
  #PMO_ORT_RELEASE_SESSION_OPTIONS = 100
EndEnumeration

#PMO_ORT_LOG_WARNING = 2
#PMO_ORT_DISABLE_ALL = 0
#PMO_ORT_ARENA_ALLOCATOR = 1
#PMO_ORT_MEMTYPE_DEFAULT = 0

Prototype.i PmoOrtGetApiBaseFn()
Prototype.i PmoOrtGetApiFn(Version.l)
Prototype.i PmoOrtGetVersionStringFn()
Prototype.i PmoOrtGetErrorMessageFn(*Status)
Prototype PmoOrtReleaseFn(*Object)
Prototype.i PmoOrtCreateEnvFn(LogLevel.l, *LogId, *Out.Integer)
Prototype.i PmoOrtCreateSessionOptionsFn(*Out.Integer)
Prototype.i PmoOrtSetGraphOptimizationFn(*Options, Level.l)
Prototype.i PmoOrtCreateSessionFromArrayFn(*Environment, *ModelData, ModelBytes.i,
                                            *Options, *Out.Integer)
Prototype.i PmoOrtCreateCpuMemoryInfoFn(AllocatorType.l, MemoryType.l, *Out.Integer)
Prototype.i PmoOrtCreateTensorWithDataFn(*MemoryInfo, *Data, DataBytes.i,
                                         *Shape, ShapeCount.i, ElementType.l,
                                         *Out.Integer)
Prototype.i PmoOrtRunFn(*Session, *RunOptions, *InputNames, *Inputs,
                        InputCount.i, *OutputNames, OutputCount.i, *Outputs)
Prototype.i PmoOrtGetTensorTypeAndShapeFn(*Value, *Out.Integer)
Prototype.i PmoOrtGetTensorElementTypeFn(*Info, *Out.Long)
Prototype.i PmoOrtGetDimensionsCountFn(*Info, *Out.Integer)
Prototype.i PmoOrtGetDimensionsFn(*Info, *Dimensions, DimensionCount.i)
Prototype.i PmoOrtGetTensorElementCountFn(*Info, *Out.Integer)
Prototype.i PmoOrtGetTensorMutableDataFn(*Value, *Out.Integer)

Structure PmoOrtSession
  *Environment
  *Options
  *Session
  *MemoryInfo
EndStructure

Global PmoOrtLibrary.i
Global *PmoOrtApi
Global PmoOrtVersion.s
Global PmoOrtError.s
Global PmoOrtGetErrorMessage.PmoOrtGetErrorMessageFn
Global PmoOrtReleaseStatus.PmoOrtReleaseFn

Procedure PmoOrtResetError()
  PmoOrtError = ""
EndProcedure

Procedure.i PmoOrtFail(Message.s)
  If PmoOrtError = ""
    PmoOrtError = Message
  EndIf
  ProcedureReturn #False
EndProcedure

Procedure.i PmoOrtFunction(Index.i)
  If *PmoOrtApi = 0 Or Index < 0
    ProcedureReturn 0
  EndIf
  ProcedureReturn PeekI(*PmoOrtApi + Index * SizeOf(Integer))
EndProcedure

Procedure.i PmoOrtCheck(*Status, Context.s)
  Protected *Message
  Protected Detail.s
  If *Status = 0
    ProcedureReturn #True
  EndIf
  If PmoOrtGetErrorMessage
    *Message = PmoOrtGetErrorMessage(*Status)
    If *Message
      Detail = PeekS(*Message, -1, #PB_UTF8)
    EndIf
  EndIf
  If PmoOrtReleaseStatus
    PmoOrtReleaseStatus(*Status)
  EndIf
  If Detail = "" : Detail = "unknown ONNX Runtime error" : EndIf
  ProcedureReturn PmoOrtFail(Context + ": " + Detail)
EndProcedure

Procedure PmoOrtUnload()
  *PmoOrtApi = 0
  PmoOrtVersion = ""
  PmoOrtGetErrorMessage = 0
  PmoOrtReleaseStatus = 0
  If PmoOrtLibrary
    CloseLibrary(PmoOrtLibrary)
    PmoOrtLibrary = 0
  EndIf
EndProcedure

Procedure.s PmoOrtDefaultLibraryName()
  CompilerSelect #PB_Compiler_OS
    CompilerCase #PB_OS_Windows
      ProcedureReturn "onnxruntime.dll"
    CompilerCase #PB_OS_Linux
      ProcedureReturn "libonnxruntime.so"
    CompilerCase #PB_OS_MacOS
      ProcedureReturn "libonnxruntime.dylib"
  CompilerEndSelect
EndProcedure

Procedure.i PmoOrtLoad(Path.s)
  Protected PmoOrtGetApiBase.PmoOrtGetApiBaseFn
  Protected PmoOrtGetApi.PmoOrtGetApiFn
  Protected PmoOrtGetVersionString.PmoOrtGetVersionStringFn
  Protected *Base
  Protected *Version
  PmoOrtUnload()
  PmoOrtResetError()
  If Path = "" : Path = PmoOrtDefaultLibraryName() : EndIf
  PmoOrtLibrary = OpenLibrary(#PB_Any, Path)
  If PmoOrtLibrary = 0
    ProcedureReturn PmoOrtFail("cannot load native ONNX Runtime library: " + Path)
  EndIf
  PmoOrtGetApiBase = GetFunction(PmoOrtLibrary, "OrtGetApiBase")
  If PmoOrtGetApiBase = 0
    PmoOrtUnload()
    ProcedureReturn PmoOrtFail("ONNX Runtime library has no OrtGetApiBase entry point")
  EndIf
  *Base = PmoOrtGetApiBase()
  If *Base = 0
    PmoOrtUnload()
    ProcedureReturn PmoOrtFail("OrtGetApiBase returned null")
  EndIf
  PmoOrtGetApi = PeekI(*Base)
  PmoOrtGetVersionString = PeekI(*Base + SizeOf(Integer))
  If PmoOrtGetApi = 0 Or PmoOrtGetVersionString = 0
    PmoOrtUnload()
    ProcedureReturn PmoOrtFail("ONNX Runtime API base is incomplete")
  EndIf
  *PmoOrtApi = PmoOrtGetApi(#PMO_ORT_API_VERSION)
  If *PmoOrtApi = 0
    PmoOrtUnload()
    ProcedureReturn PmoOrtFail("ONNX Runtime does not provide C API 29")
  EndIf
  *Version = PmoOrtGetVersionString()
  If *Version
    PmoOrtVersion = PeekS(*Version, -1, #PB_UTF8)
  EndIf
  PmoOrtGetErrorMessage = PmoOrtFunction(#PMO_ORT_GET_ERROR_MESSAGE)
  PmoOrtReleaseStatus = PmoOrtFunction(#PMO_ORT_RELEASE_STATUS)
  If PmoOrtGetErrorMessage = 0 Or PmoOrtReleaseStatus = 0
    PmoOrtUnload()
    ProcedureReturn PmoOrtFail("ONNX Runtime API 29 status functions are missing")
  EndIf
  ProcedureReturn #True
EndProcedure

Procedure PmoOrtCloseSession(*State.PmoOrtSession)
  Protected Release.PmoOrtReleaseFn
  If *State = 0 : ProcedureReturn : EndIf
  If *State\MemoryInfo
    Release = PmoOrtFunction(#PMO_ORT_RELEASE_MEMORY_INFO)
    If Release : Release(*State\MemoryInfo) : EndIf
  EndIf
  If *State\Session
    Release = PmoOrtFunction(#PMO_ORT_RELEASE_SESSION)
    If Release : Release(*State\Session) : EndIf
  EndIf
  If *State\Options
    Release = PmoOrtFunction(#PMO_ORT_RELEASE_SESSION_OPTIONS)
    If Release : Release(*State\Options) : EndIf
  EndIf
  If *State\Environment
    Release = PmoOrtFunction(#PMO_ORT_RELEASE_ENV)
    If Release : Release(*State\Environment) : EndIf
  EndIf
  *State\Environment = 0 : *State\Options = 0
  *State\Session = 0 : *State\MemoryInfo = 0
EndProcedure

Procedure.i PmoOrtOpenSession(*ModelData, ModelBytes.i, *State.PmoOrtSession)
  Protected CreateEnv.PmoOrtCreateEnvFn = PmoOrtFunction(#PMO_ORT_CREATE_ENV)
  Protected CreateOptions.PmoOrtCreateSessionOptionsFn = PmoOrtFunction(#PMO_ORT_CREATE_SESSION_OPTIONS)
  Protected SetOptimization.PmoOrtSetGraphOptimizationFn = PmoOrtFunction(#PMO_ORT_SET_GRAPH_OPTIMIZATION)
  Protected CreateSession.PmoOrtCreateSessionFromArrayFn = PmoOrtFunction(#PMO_ORT_CREATE_SESSION_FROM_ARRAY)
  Protected CreateMemory.PmoOrtCreateCpuMemoryInfoFn = PmoOrtFunction(#PMO_ORT_CREATE_CPU_MEMORY_INFO)
  Protected *LogId = UTF8("puremetal-onnx-compiler")
  Protected Result.i
  If *State = 0 Or *ModelData = 0 Or ModelBytes <= 0
    If *LogId : FreeMemory(*LogId) : EndIf
    ProcedureReturn PmoOrtFail("invalid native trace session input")
  EndIf
  PmoOrtCloseSession(*State)
  If CreateEnv = 0 Or CreateOptions = 0 Or SetOptimization = 0 Or
     CreateSession = 0 Or CreateMemory = 0
    If *LogId : FreeMemory(*LogId) : EndIf
    ProcedureReturn PmoOrtFail("ONNX Runtime API 29 session functions are missing")
  EndIf
  Result = PmoOrtCheck(CreateEnv(#PMO_ORT_LOG_WARNING, *LogId, @*State\Environment),
                       "cannot create ONNX Runtime environment")
  FreeMemory(*LogId)
  If Result = 0 : Goto PmoOrtOpenSessionFailed : EndIf
  If PmoOrtCheck(CreateOptions(@*State\Options), "cannot create ONNX Runtime session options") = 0
    Goto PmoOrtOpenSessionFailed
  EndIf
  If PmoOrtCheck(SetOptimization(*State\Options, #PMO_ORT_DISABLE_ALL),
                 "cannot disable ONNX Runtime graph rewrites") = 0
    Goto PmoOrtOpenSessionFailed
  EndIf
  If PmoOrtCheck(CreateSession(*State\Environment, *ModelData, ModelBytes,
                               *State\Options, @*State\Session),
                 "cannot create ONNX Runtime trace session") = 0
    Goto PmoOrtOpenSessionFailed
  EndIf
  If PmoOrtCheck(CreateMemory(#PMO_ORT_ARENA_ALLOCATOR, #PMO_ORT_MEMTYPE_DEFAULT,
                              @*State\MemoryInfo),
                 "cannot create ONNX Runtime CPU memory descriptor") = 0
    Goto PmoOrtOpenSessionFailed
  EndIf
  ProcedureReturn #True

  PmoOrtOpenSessionFailed:
  PmoOrtCloseSession(*State)
  ProcedureReturn #False
EndProcedure
