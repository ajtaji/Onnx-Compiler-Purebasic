"""Check fixed INT8 scratch planning; pass an explicit output directory.

Requires Python's onnx package and the standard rebuilt compiler CLI.
Generated sources are also suitable for a downstream target build.
"""
from pathlib import Path
import re
import subprocess
import sys

import onnx
from onnx import TensorProto, helper


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: py tests/conv_gather_fixed_exports.py OUTPUT_DIRECTORY")
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    compiler = Path(__file__).resolve().parents[1] / "bin/PureMetalOnnxCompilerCLI.exe"
    # Golden capacities include the arena's64-byte alignment. Grouped,
    # small-channel and desktop exports must keep their previous capacities.
    cases = [
        ("eligible", 128, 64, 7, 3, 1, "pi4", 93760),
        ("grouped", 128, 64, 7, 3, 2, "pi4", 28096),
        ("small", 64, 8, 3, 1, 1, "pi4", 15680),
        ("desktop", 128, 64, 7, 3, 1, "windows", 24064),
    ]
    for name, ic, oc, kernel, dilation, groups, target, expected in cases:
        width = 129
        pad = (kernel - 1) * dilation // 2
        graph = helper.make_graph(
            [helper.make_node("Conv", ["x", "w", "b"], ["y"],
                              pads=[pad, pad], dilations=[dilation], group=groups)],
            name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ic, width])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, oc, width])],
            [helper.make_tensor("w", TensorProto.FLOAT, [oc, ic // groups, kernel],
                                [((i * 19) % 255 - 127) / 97
                                 for i in range(oc * (ic // groups) * kernel)]),
             helper.make_tensor("b", TensorProto.FLOAT, [oc],
                                [(i - 31) / 71 for i in range(oc)])],
        )
        model_path = out / (name + ".onnx")
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), model_path)
        result = subprocess.run(
            [str(compiler), "--compile", str(model_path), "--target", target,
             "--precision", "int8", "--output", str(out / name)],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        source = (out / (name + (".pb" if target == "windows" else ".pi4"))).read_text(encoding="utf-8")
        match = re.search(r"#PMO_INT8_SCRATCH_BYTES\s*=\s*(\d+)", source)
        assert match and int(match[1]) == expected, (name, match[1] if match else None, expected)
        assert re.search(r"#PMD_NATIVE_BYTES\s*=\s*8\b", source), name
        print(name, "PASS", expected)


if __name__ == "__main__":
    main()
