"""Safety checks for the optional source-metadata cache preparer."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'prepare', Path(__file__).resolve().parents[1] / 'tools/prepare_immutable_weight_caches.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def fixture():
    text = ('#PMD_NATIVE_BYTES=8\n#PMO_INT8_BACKEND=1\n#PMO_USE_INT8=1\n'
            '#PMO_WEIGHT_FILE_BYTES=10000000\n'
            'Procedure PmModelResetRequest()\n DRelease(9)\nEndProcedure\n'
            'Procedure.i PmModelBindMemory(weights.i,weightBytes.i,arena.i,arenaBytes.i)\n')
    for tid, offset, kind, dims in [(1, 0, 33, [256, 257]), (2, 200000, 3, [20, 512, 11])]:
        count = prepare.math.prod(dims)
        text += (f'  Dt({tid})\\Data=PmModelWeights+{offset}\n'
                 f'  Dt({tid})\\Kind={kind}\n  Dt({tid})\\Count={count}\n'
                 f'  Dt({tid})\\Bytes={count*(2 if kind == 33 else 1)}\n'
                 f'  Dt({tid})\\Rank={len(dims)}\n')
        for i, d in enumerate(dims):
            text += f'  DDims({tid}*8+{i})={d}\n'
    text += ('EndProcedure\n DMatMul(8,9,1)\nDCall(2)=2\nDCall(3)=3\n'
             'DCall(4)=1\nDCall(5)=1\nDCall(6)=1\nDCall(7)=0\nDCall(8)=0\nDConv()\n')
    return text


class CachePreparationSafety(unittest.TestCase):
    def test_wide_tail_matrix_and_large_conv(self):
        p = prepare.plan(fixture())
        self.assertEqual(p['cache_bytes'], [256*256*2, 20*512*11, 5*11*8])
        text = prepare.emit(p)
        self.assertIn('PmI8PackWeight(Dt(1)\\Data,256,257,1)', text)
        self.assertIn('PmConvPackedCacheWeight(Dt(2)\\Data,512,20,11)', text)

    def test_released_weight_refused(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            prepare.plan(fixture().replace('DRelease(9)', 'DRelease(1)'))

    def test_out_of_pack_refused(self):
        with self.assertRaisesRegex(ValueError, 'extent'):
            prepare.plan(fixture().replace('PmModelWeights+200000', 'PmModelWeights+9999999'))

    def test_missing_dimension_not_borrowed_from_next_tensor(self):
        with self.assertRaisesRegex(ValueError, 'dimension'):
            prepare.plan(fixture().replace('  DDims(1*8+1)=257\n', ''))

    def test_corrupt_count_refused(self):
        with self.assertRaisesRegex(ValueError, 'extent'):
            prepare.plan(fixture().replace('\\Count=65792', '\\Count=65793'))

    def test_temporary_operand_not_selected(self):
        p = prepare.plan(fixture().replace('DMatMul(8,9,1)', 'DMatMul(8,1,9)'))
        self.assertEqual(p['matrix'], [])
        self.assertEqual(len(p['conv']), 1)

    def test_wrong_backend_refused(self):
        with self.assertRaisesRegex(ValueError, 'AArch64'):
            prepare.plan(fixture().replace('#PMO_INT8_BACKEND=1', '#PMO_INT8_BACKEND=0'))

    def test_float_only_model_refused(self):
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            prepare.plan(fixture().replace('\\Kind=33', '\\Kind=1').replace('\\Kind=3\n', '\\Kind=1\n'))


if __name__ == '__main__':
    unittest.main()
