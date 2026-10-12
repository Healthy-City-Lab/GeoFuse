"""``geofuse hpc script``.

The scripts give each dependency level and resource group one SLURM array of
the right size, repeat the command-line settings, chain the levels with
``afterok`` and use Unix line endings.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from test_cli_arrays import _Case


class ScriptTests(_Case):
    def test_one_array_per_level_and_group(self):
        folder = os.path.join(self.dir, "hpc")
        setup = "module load cuda; conda activate geofuse"
        flags = ["-o", folder, "--account", "acct", "--setup", setup]
        flags += ["--cloud-max", "30"]
        code, out = self.geofuse("hpc", "script", self.run, *flags)
        self.assertEqual(code, 0, out)
        names = sorted(os.listdir(folder))
        expected = ["level0-gpu.sbatch", "level0-light.sbatch", "level1-light.sbatch"]
        self.assertEqual(names, expected + ["submit.sh"])
        with open(os.path.join(folder, "level0-gpu.sbatch"), "rb") as f:
            gpu = f.read().decode()
        self.assertNotIn("\r", gpu)
        for part in (
            "--array=0-1",
            "--gres=gpu:1",
            "--account=acct",
            "module load cuda",
            "--level 0 --group gpu --cloud-max 30",
            '--array-task "${SLURM_ARRAY_TASK_ID}"',
        ):
            self.assertIn(part, gpu)
        with open(os.path.join(folder, "level0-light.sbatch")) as f:
            light = f.read()
        self.assertIn("--array=0-0", light)
        self.assertNotIn("--gres", light)
        with open(os.path.join(folder, "submit.sh")) as f:
            submit = f.read()
        level1 = submit.index("# level 1")
        self.assertLess(submit.index("level0-gpu.sbatch"), level1)
        self.assertLess(submit.index("level0-light.sbatch"), level1)
        self.assertLess(level1, submit.index("level1-light.sbatch"))
        self.assertIn("--dependency=afterok:$after", submit)

if __name__ == "__main__":
    unittest.main()
