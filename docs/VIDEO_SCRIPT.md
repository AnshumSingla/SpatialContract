# 3-minute video script

Read the **quoted text** straight through, top to bottom: about 400 words, roughly 2:40-2:50 at a calm pace (hard limit 3:00). The *italic lines* are screen cues; do not read them. Screen recording only, **no music** and no logos. Upload to YouTube as **Public** with the repo link and the demo link in the description.

Every number you say is on screen at that moment. Read them as written; do not round up.

## Prepare before recording
1. **Browser tab 1:** https://anshumsingla.github.io/SpatialContract/ (the demo). Window about 1280x720, zoom 110%. The page auto-plays the first chip 3.5 s after it loads: click **❚❚ Pause** straight away (and again whenever you click that chip).
2. **Browser tab 2:** https://github.com/AnshumSingla/SpatialContract, scrolled to the README table **"What we found"**.
3. **Terminal**, large font (18 pt+), in `D:\VS\Nebius\SpatialContract`, venv active (`.venv\Scripts\activate`). Run these **before** recording and keep the output on screen:
   - `python scripts/localise_real.py torchio_crop` (about 65 s): prints `of 101` and a ranked list whose line 6 is `Crop.apply_transform ... <- edited by the upstream fix`.
   - In a second terminal: `spatialcontract check examples/inmemory_example.py::crop_buggy examples/inmemory_example.py::crop_fixed`: prints `FAIL  worst landmark 25.3 mm off`, a `pattern:` line ending "the shift is a whole number of output voxels", and `PASS  worst landmark 0.00 mm off`.

## Script

**1. The bug (0:00-0:20)**

*Tab 1. Click the chip "Real library bug: TorchIO RAS/LPS (a different fix than upstream)". Keep "Before: library at the pre-fix commit". Drag the 3D view once; point at a blue ring, its orange dot, and the 123.3 mm label. Then point at pill 1, "✔ all pass, 87 covering tests".*

> "This is SpatialContract. This bug shipped in TorchIO, a popular medical-imaging library. Every image it saved through SimpleITK landed in the wrong place, up to a hundred and twenty-three millimetres away. Nothing crashed, and all eighty-seven of the library's own tests for this code passed."

**2. The contract and the oracle (0:20-0:55)**

*Click the first chip, "Unit tests pass. The landmarks are 100+ mm off." (pause autoplay). Highlight the `SPATIAL_CONTRACT` block in the source. Point at pill 1 "✔ all pass", then pill 2 "worst error 227.05 mm", then the landmark table.*

> "SpatialContract makes the intent explicit. A pipeline declares a contract: which world frame, which units, which orientation. A deterministic oracle then pushes a phantom, with landmarks at known physical positions, through the code, and measures in millimetres where each one lands. This script passes its unit tests, and it is two hundred and twenty-seven millimetres off."

**3. Nemotron repairs it (0:55-1:35)**

*Click ▶ Play, or step with ▶: iteration 1 ERROR, iteration 2 FAIL 83.31 mm, iteration 3 ERROR, iteration 4 PASS 0.00 mm. Show the green diamonds landing in the blue rings. Scroll to "Checks the model never saw": four ✔ tiles.*

> "Then NVIDIA Nemotron 3 Super, running on Nebius Token Factory, takes over. It reads the code and the oracle's measurements, and returns a patch as a tool call. The first patch crashes. The second is still eighty-three millimetres off. The third crashes. On the fourth, every landmark is back on target. Then the fix faces checks the model never saw: hidden phantoms, a fresh phantom, the same image stored in another orientation, and an ordinary image checked with SimpleITK."

**4. Real library bugs (1:35-2:05)**

*Terminal: the `localise_real.py` output; highlight "of 101" and line 6, "edited by the upstream fix". Then tab 1: the TorchIO RAS/LPS chip, click "After: with the agent's edit", show the diff.*

> "We also reproduced eight real bugs from TorchIO, nibabel, MONAI and nilearn. Their own tests caught none of them; the oracle catches all eight. Inside a library, a runtime trace ranks the functions that ran. For this TorchIO crop bug, a hundred and one functions ran, and the one the real fix edited ranks sixth. Across our repair methods, Nemotron produced verified fixes for four of the eight, including both of these TorchIO bugs. This fix differs from upstream, and it still passes every hidden check."

**5. Results (2:05-2:35)**

*Tab 2: the README rows "Seeded bugs: which feedback repairs more?" and "Does it leave correct code alone and refuse to guess?". Then tab 1: chip "No contract: it will not guess a convention", pill 3 "abstained". Then click "See it on a real brain scan" at the top of the page: click **Buggy output** (orange slices off the faint input, "correlation 0.68"), then **Nemotron's repair** (green, "correlation 1.00"); drag to rotate once.*

> "On thirty-one seeded bugs, Nemotron with oracle feedback repaired ninety-eight percent. With a strong hand-written test suite, ninety-seven. The contract gives that signal from just four fields, and it is the only check that can say 'unknown': where no convention could be known, the agent never guessed. And the repairs hold on real brain scans. On this one, the bug puts the anatomy in the wrong place, and Nemotron's repair lands exactly on the input."

**6. Use it today (2:35-2:55)**

*Terminal: the `spatialcontract check` output: FAIL 25.3 mm, the `pattern:` line, PASS 0.00 mm. Then tab 1: click "Try it yourself" at the top and hold on the steps. End card: the repo URL and the demo URL as text.*

> "You can use the oracle on its own today, as a decorator, a pytest fixture, or a command in CI. Here it finds a crop twenty-five millimetres off, says the shift is a whole number of voxels, and passes the fixed version. SpatialContract: Nemotron fixes spatial bugs, and every fix is checked against physical ground truth. Links are below."

## Rules for the take
- Say "seeded" when quoting 98%: those are our own bugs. Real bugs: detected 8 of 8; verified repairs for 4 of the 8 across all methods (the tool-calling agent alone: 3 of the 8). Never round up.
- If the take runs long, cut "Here it finds a crop ... the fixed version." from part 6, never part 5.
- Check on YouTube: visibility Public, no copyrighted audio, length under 3:00, repo and demo URLs in the description.
