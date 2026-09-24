; ======================================================================
; tensor_pool_windows.pbi - the worker pool of the Windows model runtime.
; ----------------------------------------------------------------------
; THE POOL SERVES THE BOUND MODEL. PmPoolStart runs when the model is bound
; (PmModelInitialize, PmOnnxBindMemory): it takes the workers it needs,
; creating any that do not exist yet. PmPoolStop runs when it is unbound
; (PmModelClose, PmOnnxUnbindMemory): every worker is parked - blocked on its
; semaphore, running nothing, holding no per-worker buffer - until the next
; bind takes it again. No operator creates a thread, and no worker ever runs
; outside a bound model's operator. A worker that is not running a task is
; blocked; it never spins (blocking at once measured as fast as spinning 10
; or 50 us between operators). Why workers are parked rather than ended is
; at PmPoolStop.
;
; A PROGRAM MAY END WITHOUT CLOSING ITS MODEL - End on an error path does -
; and End releases the language's memory while the workers still exist. Two
; things made that fault, and both are closed: a worker still being started
; by the language when End ran (bind returns only once every worker has
; started; without that wait 29 of 60 such exits faulted), and a worker
; spinning on slots in the language's memory (58 of 60). The slots and the
; control block are operating-system memory, and a worker reports a job
; finished only once it has announced it is blocking.
;
; HOW MANY THREADS. Decided when the model is bound, on the machine it runs
; on: the number of logical processors this PROCESS may use - the affinity
; mask it was started with (start /affinity, a job object's affinity, a
; parent's mask), not the machine total - and on a machine with more than
; one processor group, every active processor of every group when the
; process has not been narrowed to part of one. The processor count a
; process's default CPU set names, when one is set, lowers it again.
; #PMO_THREADS (the compiler's --threads option) and PmPoolSetThreads (the
; model's run-time setting) can only LOWER that number, never raise it.
; 1 is the single-threaded runtime: no worker is created at all.
; A job object's CPU-rate cap limits processor TIME, not which processors
; run the threads, so it needs nothing here: the workers still use only
; the permitted processors and simply receive less time.
;
; SAME BITS ON EVERY THREAD COUNT. An operator is split into TASKS, and a
; task owns a disjoint range of OUTPUT elements, rows or channels. Every
; output value is computed by exactly one task with exactly the arithmetic,
; operand order and summation order of the single-threaded kernel, so the
; thread count changes who computes a value and never what it is. No
; reduction is ever split. Workers share only read-only inputs and disjoint
; output slices; per-thread scratch belongs to the operator call and is
; indexed by the worker number a task receives. Each worker loads the
; caller's x87 control word and MXCSR before its tasks, so the floating-
; point environment is the caller's.
;
; SMALL OPERATORS STAY ON THE CALLING THREAD. PmPoolTasks returns 1 below
; an operator's grain (the smallest task worth a hand-off, measured on the
; reference machine - docs/VALIDATION.md), and a single task never touches
; the pool.
;
; NO RECURSION, AND NO NESTING: a task that reaches another parallel
; operator runs it inline on its own thread (the pool is marked busy for
; the duration of a run), and so does a second thread calling in while a
; run is in progress.
; ======================================================================

CompilerIf Defined(PMO_THREADS, #PB_Constant) = 0
  #PMO_THREADS = 0
CompilerEndIf

Prototype PmPoolTaskProc(*ctx, task.i, worker.i)
Prototype.i PmPoolApiNone()
Prototype.i PmPoolApiWord(value.i)
Prototype.i PmPoolApiGroups(process.i, *count, *groups)
Prototype.i PmPoolApiCpuSets(process.i, *ids, count.i, *required)
Prototype.i PmPoolApiGroupAffinity(thread.i, *affinity, *previous)

; Control block, one cache line per field that more than one thread writes.
#PMPOOL_NEXT = 0        ; next task number (lock xadd)
#PMPOOL_PENDING = 64    ; helpers still running the current job
#PMPOOL_BUSY = 128      ; 1 while a run is in progress
#PMPOOL_PROC = 192      ; job descriptor, written by the caller only
#PMPOOL_CTX = 200
#PMPOOL_TASKS = 208
#PMPOOL_CW = 216
#PMPOOL_MXCSR = 224
#PMPOOL_QUIT = 232
#PMPOOL_HELPERS = 240   ; helpers woken for the current job
#PMPOOL_STARTED = 248   ; workers that have begun running
#PMPOOL_CONTROL_BYTES = 256
; One 64-byte slot per thread; slot 0 is the calling thread.
#PMPOOL_SLOT_BYTES = 64
#PMPOOL_THREAD = 0
#PMPOOL_GO = 8
#PMPOOL_TICKET = 16
#PMPOOL_SLEEPING = 24
#PMPOOL_GROUP = 32
#PMPOOL_SCRATCH = 40
#PMPOOL_SCRATCH_BYTES = 48

Global PmPoolThreads.i = 1
Global PmPoolRequested.i = #PMO_THREADS
Global PmPoolAllowed.i = 1
Global PmPoolRunning.i
Global PmPoolBound.i
; Workers created so far (slots 1 .. PmPoolCreated); they are parked, never ended.
Global PmPoolCreated.i
#PMPOOL_MOST = 4096
Global *PmPoolControl
Global *PmPoolSlots
Global *PmPoolControlMemory
Global *PmPoolSlotMemory
; The control block and the slots are operating-system memory (VirtualAlloc),
; not the host language's, so nothing a worker touches between reporting its
; job done and blocking again is released by End (see the header).
#PMPOOL_MEM_COMMIT = $3000
#PMPOOL_PAGE_READWRITE = $04
#PMPOOL_MEM_RELEASE = $8000
; Test hook for the split gate: 1 splits every splittable operator into
; the smallest tasks it allows, so small odd shapes still split unevenly.
Global PmPoolForceSplit.i
; Runs that handed tasks to at least one worker (read by the split gate).
Global PmPoolSplitRuns.i
; Worker 0's (the calling thread's) own working buffer; the others' are in
; their slots. See PmPoolScratch.
Global PmPoolScratch0.i, PmPoolScratch0Bytes.i
; Called (when set) by PmPoolStop before the workers are joined, so a layer
; above the pool can hand back memory it kept for the bound model.
Prototype PmPoolHookProc()
Global PmPoolStopHook.PmPoolHookProc

Procedure.i PmPoolXadd(*address, value.i)
  Protected pa.i = *address, v.i = value
  !mov rcx,[p.v_pa]
  !mov rax,[p.v_v]
  !lock xadd [rcx],rax
  !mov [p.v_v],rax
  ProcedureReturn v
EndProcedure

Procedure.i PmPoolXchg(*address, value.i)
  Protected pa.i = *address, v.i = value
  !mov rcx,[p.v_pa]
  !mov rax,[p.v_v]
  !xchg [rcx],rax
  !mov [p.v_v],rax
  ProcedureReturn v
EndProcedure

Procedure.i PmPoolMax(a.i, b.i)
  If a > b : ProcedureReturn a : EndIf
  ProcedureReturn b
EndProcedure

Procedure.i PmPoolPopCount(mask.q)
  Protected n.i
  While mask
    mask = mask & (mask - 1)
    n + 1
  Wend
  ProcedureReturn n
EndProcedure

; The allowed-processor rule, apart from the system calls so a gate can
; drive it with any machine shape:
;   groups        active processor groups
;   *active       active processors per group (groups native integers)
;   processMask   GetProcessAffinityMask's process mask; 0 when the call
;                 reports the process spans several groups
;   primary       the group the process started in
;   cpuSetCount   processors in the process's default CPU set, 0 for none
; A mask narrower than its whole group is an explicit restriction and is
; the answer; a whole group on a multi-group machine is not a restriction,
; and every active processor of every group may be used.
Procedure.i PmPoolAllowedFrom(groups.i, *active, processMask.q, primary.i, cpuSetCount.i)
  Protected n.i, g.i, whole.i
  If groups < 1 : groups = 1 : EndIf
  If primary < 0 Or primary >= groups : primary = 0 : EndIf
  whole = PeekI(*active + primary * 8)
  If processMask <> 0 And (groups = 1 Or PmPoolPopCount(processMask) < whole)
    n = PmPoolPopCount(processMask)
  Else
    For g = 0 To groups - 1 : n + PeekI(*active + g * 8) : Next
  EndIf
  If cpuSetCount > 0 And cpuSetCount < n : n = cpuSetCount : EndIf
  If n < 1 : n = 1 : EndIf
  ProcedureReturn n
EndProcedure

; The processors this process may use, from the running system.
; *groupsOut (optional) receives the active group count, and *activeOut
; (optional, 64 native integers) each group's active processor count.
Procedure.i PmPoolDetect(*groupsOut = 0, *activeOut = 0, *primaryOut = 0)
  Protected processMask.i, systemMask.i, groups.i = 1, g.i, primary.i, cpuSets.i, required.l, lib.i, n.i
  Protected groupCount.PmPoolApiNone, activeCount.PmPoolApiWord, processGroups.PmPoolApiGroups, defaultSets.PmPoolApiCpuSets
  Protected Dim active.i(63), Dim groupList.u(63)
  Protected listed.u = 64
  If GetProcessAffinityMask_(GetCurrentProcess_(), @processMask, @systemMask) = 0
    ProcedureReturn CountCPUs(#PB_System_ProcessCPUs)
  EndIf
  active(0) = PmPoolPopCount(systemMask)
  lib = OpenLibrary(#PB_Any, "kernel32.dll")
  If lib
    groupCount = GetFunction(lib, "GetActiveProcessorGroupCount")
    activeCount = GetFunction(lib, "GetActiveProcessorCount")
    processGroups = GetFunction(lib, "GetProcessGroupAffinity")
    defaultSets = GetFunction(lib, "GetProcessDefaultCpuSets")
    If groupCount And activeCount
      groups = groupCount() & $FFFF
      If groups < 1 : groups = 1 : EndIf
      If groups > 64 : groups = 64 : EndIf
      For g = 0 To groups - 1 : active(g) = activeCount(g) : Next
    EndIf
    If groups > 1 And processGroups
      If processGroups(GetCurrentProcess_(), @listed, @groupList(0)) And listed >= 1
        primary = groupList(0)
      EndIf
    EndIf
    If defaultSets
      If defaultSets(GetCurrentProcess_(), 0, 0, @required) = 0 : cpuSets = required : EndIf
    EndIf
    CloseLibrary(lib)
  EndIf
  If *groupsOut : PokeI(*groupsOut, groups) : EndIf
  If *primaryOut : PokeI(*primaryOut, primary) : EndIf
  If *activeOut
    For g = 0 To groups - 1 : PokeI(*activeOut + g * 8, active(g)) : Next
  EndIf
  n = PmPoolAllowedFrom(groups, @active(0), processMask, primary, cpuSets)
  ProcedureReturn n
EndProcedure

Declare PmPoolWake(i.i)

Procedure PmPoolWorker(index.i)
  Protected slot.i = *PmPoolSlots + index * #PMPOOL_SLOT_BYTES, control.i = *PmPoolControl
  Protected seen.i, task.i, tasks.i, cw.i, mx.i, ctx.i, helpers.i, child.i, working.i
  Protected run.PmPoolTaskProc
  ; Tickets start at 0 in freshly zeroed slots. Never read the first one:
  ; a caller may already have raised it before this thread began running.
  seen = 0
  Repeat
    ; Announce the sleep, and only then report the last job finished (or,
    ; the first time, this thread started): the caller can go on - even to
    ; End - only once the worker is about to block. Then look once more: a
    ; caller that raised the ticket after the exchange sees Sleeping = 1 and
    ; signals; one that raised it before is seen by the second look.
    PmPoolXchg(slot + #PMPOOL_SLEEPING, 1)
    If working
      PmPoolXadd(control + #PMPOOL_PENDING, -1)
    Else
      PmPoolXadd(control + #PMPOOL_STARTED, 1)
    EndIf
    If PeekI(slot + #PMPOOL_TICKET) = seen
      WaitSemaphore(PeekI(slot + #PMPOOL_GO))
    ElseIf PmPoolXchg(slot + #PMPOOL_SLEEPING, 0) = 0
      ; The caller took the flag first and has signalled: consume it.
      WaitSemaphore(PeekI(slot + #PMPOOL_GO))
    EndIf
    seen = PeekI(slot + #PMPOOL_TICKET)
    If PeekI(control + #PMPOOL_QUIT) : Break : EndIf
    cw = PeekI(control + #PMPOOL_CW) : mx = PeekI(control + #PMPOOL_MXCSR)
    !fldcw word [p.v_cw]
    !ldmxcsr dword [p.v_mx]
    run = PeekI(control + #PMPOOL_PROC) : ctx = PeekI(control + #PMPOOL_CTX) : tasks = PeekI(control + #PMPOOL_TASKS)
    ; The wake-up is a tree: helper k wakes helpers 2k+1 and 2k+2, so the
    ; last of 31 is running after five hand-offs instead of thirty-one.
    helpers = PeekI(control + #PMPOOL_HELPERS)
    child = 2 * index + 1
    If child <= helpers : PmPoolWake(child) : EndIf
    If child + 1 <= helpers : PmPoolWake(child + 1) : EndIf
    Repeat
      task = PmPoolXadd(control + #PMPOOL_NEXT, 1)
      If task >= tasks : Break : EndIf
      run(ctx, task, index)
    ForEver
    working = 1
  ForEver
EndProcedure

; Wakes the helper in slot i for the job just published.
Procedure PmPoolWake(i.i)
  Protected slot.i = *PmPoolSlots + i * #PMPOOL_SLOT_BYTES
  PokeI(slot + #PMPOOL_TICKET, PeekI(slot + #PMPOOL_TICKET) + 1)
  If PmPoolXchg(slot + #PMPOOL_SLEEPING, 0)
    SignalSemaphore(PeekI(slot + #PMPOOL_GO))
  EndIf
EndProcedure

; A worker's own working buffer of at least `bytes`, not zeroed, kept from
; one operator to the next (a fresh block's first touch is a page fault the
; system serves one at a time, however many workers ask) and freed when the
; pool stops. Only worker `worker` itself may call it, from the task it was
; given - never from an operator a task runs inline. 0 when memory runs out.
Procedure.i PmPoolScratch(worker.i, bytes.i)
  Protected slot.i, *m
  If worker = 0 Or *PmPoolSlots = 0
    If PmPoolScratch0Bytes < bytes
      If PmPoolScratch0 : FreeMemory(PmPoolScratch0) : EndIf
      PmPoolScratch0 = AllocateMemory(bytes + 64, #PB_Memory_NoClear) : PmPoolScratch0Bytes = 0
      If PmPoolScratch0 : PmPoolScratch0Bytes = bytes : EndIf
    EndIf
    ProcedureReturn PmPoolScratch0
  EndIf
  slot = *PmPoolSlots + worker * #PMPOOL_SLOT_BYTES
  If PeekI(slot + #PMPOOL_SCRATCH_BYTES) < bytes
    If PeekI(slot + #PMPOOL_SCRATCH) : FreeMemory(PeekI(slot + #PMPOOL_SCRATCH)) : EndIf
    *m = AllocateMemory(bytes + 64, #PB_Memory_NoClear)
    PokeI(slot + #PMPOOL_SCRATCH, *m) : PokeI(slot + #PMPOOL_SCRATCH_BYTES, 0)
    If *m : PokeI(slot + #PMPOOL_SCRATCH_BYTES, bytes) : EndIf
  EndIf
  ProcedureReturn PeekI(slot + #PMPOOL_SCRATCH)
EndProcedure

; Unbind: every worker is PARKED - blocked on its semaphore, running
; nothing - and the per-worker buffers are freed; the next bind reuses the
; parked workers (and creates only the ones it lacks). Workers are never
; made to exit: a thread of this runtime that has run procedures leaves
; work for the system's own thread-pool workers when it exits, and a
; program that Ends soon after races them - with fault dialogs switched
; off (SEM_NOGPFAULTERRORBOX, as a test harness sets) the exit reported an
; access violation on 89 of 600 runs of a twelve-line program with no pool
; code in it, and a one-tick pause after the join still left 1 in 400. A
; program that Ends with its threads blocked was clean every time. A parked
; worker holds a stack and a semaphore and uses no processor time.
; Safe to call when no pool is running.
Procedure PmPoolStop()
  Protected i.i, slot.i
  PmPoolBound = 0
  If PmPoolStopHook : PmPoolStopHook() : EndIf
  If PmPoolScratch0 : FreeMemory(PmPoolScratch0) : PmPoolScratch0 = 0 : PmPoolScratch0Bytes = 0 : EndIf
  For i = 1 To PmPoolCreated
    slot = *PmPoolSlots + i * #PMPOOL_SLOT_BYTES
    If PeekI(slot + #PMPOOL_SCRATCH)
      FreeMemory(PeekI(slot + #PMPOOL_SCRATCH)) : PokeI(slot + #PMPOOL_SCRATCH, 0) : PokeI(slot + #PMPOOL_SCRATCH_BYTES, 0)
    EndIf
  Next
  PmPoolThreads = 1 : PmPoolRunning = 0
EndProcedure

; Bind: the allowed processor count, lowered by PmPoolRequested when that
; is set and smaller. Parked workers are taken first; missing ones are
; created (the slots have room for #PMPOOL_MOST). Returns the thread count.
; A worker that cannot be created lowers the count; it is never an error.
Procedure.i PmPoolStart()
  Protected n.i, i.i, slot.i, t.i, g.i, k.i, used.i, groups.i = 1, primary.i, lib.i
  Protected Dim active.i(63)
  Protected groupAffinity.PmPoolApiGroupAffinity
  Protected Dim affinity.q(1)
  PmPoolStop()
  PmPoolBound = 1
  PmPoolAllowed = PmPoolDetect(@groups, @active(0), @primary)
  n = PmPoolAllowed
  If PmPoolRequested > 0 And PmPoolRequested < n : n = PmPoolRequested : EndIf
  If n > #PMPOOL_MOST : n = #PMPOOL_MOST : EndIf
  If n < 1 : n = 1 : EndIf
  If n = 1 : PmPoolThreads = 1 : ProcedureReturn 1 : EndIf
  If *PmPoolControl = 0
    ; Zero-filled operating-system pages (see the note at the globals), kept
    ; for the process's life because parked workers point into them.
    *PmPoolControlMemory = VirtualAlloc_(0, #PMPOOL_CONTROL_BYTES + 64, #PMPOOL_MEM_COMMIT, #PMPOOL_PAGE_READWRITE)
    *PmPoolSlotMemory = VirtualAlloc_(0, (#PMPOOL_MOST + 1) * #PMPOOL_SLOT_BYTES + 64, #PMPOOL_MEM_COMMIT, #PMPOOL_PAGE_READWRITE)
    If *PmPoolControlMemory = 0 Or *PmPoolSlotMemory = 0
      If *PmPoolControlMemory : VirtualFree_(*PmPoolControlMemory, 0, #PMPOOL_MEM_RELEASE) : EndIf
      If *PmPoolSlotMemory : VirtualFree_(*PmPoolSlotMemory, 0, #PMPOOL_MEM_RELEASE) : EndIf
      *PmPoolControlMemory = 0 : *PmPoolSlotMemory = 0 : PmPoolThreads = 1
      ProcedureReturn 1
    EndIf
    *PmPoolControl = (*PmPoolControlMemory + 63) & ~63
    *PmPoolSlots = (*PmPoolSlotMemory + 63) & ~63
  EndIf
  ; With several processor groups a worker is placed in the next group once
  ; the calling thread's group is full (a thread starts in its creator's).
  If groups > 1
    lib = OpenLibrary(#PB_Any, "kernel32.dll")
    If lib : groupAffinity = GetFunction(lib, "SetThreadGroupAffinity") : EndIf
  EndIf
  For i = PmPoolCreated + 1 To n - 1
    slot = *PmPoolSlots + i * #PMPOOL_SLOT_BYTES
    PokeI(slot + #PMPOOL_GO, CreateSemaphore(0))
    t = 0
    If PeekI(slot + #PMPOOL_GO) : t = CreateThread(@PmPoolWorker(), i) : EndIf
    If t = 0
      If PeekI(slot + #PMPOOL_GO) : FreeSemaphore(PeekI(slot + #PMPOOL_GO)) : PokeI(slot + #PMPOOL_GO, 0) : EndIf
      Break
    EndIf
    PokeI(slot + #PMPOOL_THREAD, t)
    PmPoolCreated = i
    If groups > 1 And groupAffinity
      ; groups in the order primary, then the others ascending
      k = 0 : g = primary : used = i
      While k < groups - 1 And used >= active(g)
        used - active(g) : k + 1
        g = k - 1 : If g >= primary : g + 1 : EndIf
      Wend
      If g <> primary And active(g) > 0
        affinity(0) = -1 : If active(g) < 64 : affinity(0) = (1 << active(g)) - 1 : EndIf
        affinity(1) = g
        groupAffinity(ThreadID(t), @affinity(0), 0)
      EndIf
      PokeI(slot + #PMPOOL_GROUP, g)
    EndIf
  Next
  If lib : CloseLibrary(lib) : EndIf
  ; Bind returns only once every worker has started and is about to block,
  ; so a program that ends right after binding never ends under a worker
  ; that is still starting.
  While PeekI(*PmPoolControl + #PMPOOL_STARTED) < PmPoolCreated
    !pause
    Delay(0)
  Wend
  PmPoolThreads = n : If PmPoolThreads > PmPoolCreated + 1 : PmPoolThreads = PmPoolCreated + 1 : EndIf
  PmPoolRunning = Bool(PmPoolThreads > 1)
  ProcedureReturn PmPoolThreads
EndProcedure
; The run-time setting: 0 or less means every allowed processor. When a
; model is bound the pool is rebuilt at once; otherwise the next bind uses
; it. Returns the thread count now in use (1 while nothing is bound).
Procedure.i PmPoolSetThreads(threads.i)
  If threads < 0 : threads = 0 : EndIf
  PmPoolRequested = threads
  If PmPoolBound
    ProcedureReturn PmPoolStart()
  EndIf
  ProcedureReturn PmPoolThreads
EndProcedure

Procedure.i PmPoolThreadCount()
  ProcedureReturn PmPoolThreads
EndProcedure

; Runs task 0 .. tasks-1 of `proc` over `ctx` and returns when every one
; has finished. The caller runs tasks too, as worker 0.
Procedure PmPoolRun(proc.i, *ctx, tasks.i)
  Protected run.PmPoolTaskProc = proc, helpers.i, i.i, task.i, control.i, cw.i, mx.i, spins.i
  If tasks <= 0 : ProcedureReturn : EndIf
  control = *PmPoolControl
  If tasks = 1 Or PmPoolThreads <= 1 Or control = 0
    For task = 0 To tasks - 1 : run(*ctx, task, 0) : Next
    ProcedureReturn
  EndIf
  If PmPoolXchg(control + #PMPOOL_BUSY, 1) <> 0
    For task = 0 To tasks - 1 : run(*ctx, task, 0) : Next
    ProcedureReturn
  EndIf
  helpers = PmPoolThreads - 1 : If helpers > tasks - 1 : helpers = tasks - 1 : EndIf
  PmPoolSplitRuns + 1
  !fnstcw word [p.v_cw]
  !stmxcsr dword [p.v_mx]
  PokeI(control + #PMPOOL_PROC, proc) : PokeI(control + #PMPOOL_CTX, *ctx) : PokeI(control + #PMPOOL_TASKS, tasks)
  PokeI(control + #PMPOOL_CW, cw & $FFFF) : PokeI(control + #PMPOOL_MXCSR, mx & $FFFFFFFF)
  PokeI(control + #PMPOOL_NEXT, 0) : PokeI(control + #PMPOOL_PENDING, helpers)
  PokeI(control + #PMPOOL_HELPERS, helpers)
  PmPoolWake(1)
  If helpers >= 2 : PmPoolWake(2) : EndIf
  Repeat
    task = PmPoolXadd(control + #PMPOOL_NEXT, 1)
    If task >= tasks : Break : EndIf
    run(*ctx, task, 0)
  ForEver
  ; THE JOIN: no output is read, and no scratch freed, before every helper
  ; has finished its last task.
  While PeekI(control + #PMPOOL_PENDING) > 0
    !pause
    spins + 1
    If (spins & 4095) = 0 : Delay(0) : EndIf
  Wend
  PokeI(control + #PMPOOL_BUSY, 0)
EndProcedure

; How many tasks to cut `count` units of work into, each at least `grain`
; units (the measured break-even) and a multiple of `align`; the chunk
; size goes to *chunk. 1 task (the calling thread alone) below two grains
; or with one thread. Tasks outnumber threads so a slower core simply
; takes fewer of them; each still exceeds the grain.
Procedure.i PmPoolTasks(count.i, grain.i, align.i, *chunk.Integer)
  Protected tasks.i, chunk.i, most.i
  If align < 1 : align = 1 : EndIf
  If grain < align : grain = align : EndIf
  *chunk\i = count
  If count <= 0 : ProcedureReturn 0 : EndIf
  If PmPoolThreads <= 1 : ProcedureReturn 1 : EndIf
  If PmPoolForceSplit
    chunk = align
    most = 4 * PmPoolThreads + 3
    If (count + chunk - 1) / chunk > most
      chunk = ((count + most - 1) / most + align - 1) / align * align
    EndIf
  Else
    If count < 2 * grain : ProcedureReturn 1 : EndIf
    most = 8 * PmPoolThreads
    chunk = (count + most - 1) / most
    If chunk < grain : chunk = grain : EndIf
    chunk = (chunk + align - 1) / align * align
  EndIf
  tasks = (count + chunk - 1) / chunk
  *chunk\i = chunk
  ProcedureReturn tasks
EndProcedure

; ---- zero-filled memory, filled by the pool ----------------------------------
; A reused block is zeroed in parallel ranges before it is handed out again.
Structure PmPoolZeroJob
  Base.i
  Bytes.i
  Chunk.i
EndStructure

Procedure PmPoolZeroTask(*j.PmPoolZeroJob, task.i, worker.i)
  Protected first.i = task * *j\Chunk, n.i = *j\Bytes - first
  If n > *j\Chunk : n = *j\Chunk : EndIf
  If n > 0 : FillMemory(*j\Base + first, n, 0) : EndIf
EndProcedure

#PMPOOL_ZERO_GRAIN = 262144

; Zeroes `bytes` at *mem, in parallel ranges when it is large.
Procedure PmPoolZeroFill(*mem, bytes.i)
  Protected j.PmPoolZeroJob, tasks.i
  j\Base = *mem : j\Bytes = bytes
  tasks = PmPoolTasks(bytes, #PMPOOL_ZERO_GRAIN, 4096, @j\Chunk)
  If tasks <= 1
    FillMemory(*mem, bytes, 0)
  Else
    PmPoolRun(@PmPoolZeroTask(), @j, tasks)
  EndIf
EndProcedure