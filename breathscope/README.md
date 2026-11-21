# breathscope

Estimate a person's respiratory rate from ordinary video and *show* where the
evidence comes from. Handles a panning / zooming camera.

```bash
uv run breathscope.py clip.mp4                      # writes a gridded reference frame, then exits
uv run breathscope.py clip.mp4 --roi 310,190,680,470   # X,Y,W,H around head + shoulders + torso
uv run breathscope.py clip.mp4 --pick               # or draw the box with the mouse (local display)
```

Dependencies are declared inline (PEP 723), so `uv run` needs no setup.
`ffmpeg` on the PATH is optional (browser-playable H.264 output).

## Outputs (`<video>_breathscope/`)

| File | What it is |
|---|---|
| `report.png` | Per-point motion map, per-point correlation map, the breathing trace, and the spectrum against the static-scene noise floor |
| `magnified.mp4` | Stabilised original next to an Eulerian motion-magnified copy, with the trace scrubbing underneath |
| `result.json` | All numbers: three rate estimators, amplitudes, resolution limit, cycle count, caveats |
| `reference_frame.png` | First frame of the analysed shot with a pixel grid and the ROI |

## How it works

1. **Shot detection** - one continuous shot is analysed (longest by default, `--shot N` to choose).
2. **Tracking** - Shi-Tomasi corners tracked with pyramidal Lucas-Kanade, always
   reference -> frame *t* with a forward-backward check, so error cannot
   accumulate into slow drift (the band breathing lives in).
3. **Camera removal** - per-frame RANSAC similarity transform fitted to points
   *outside* the ROI. The remainder is motion relative to the static scene.
4. **Signal** - ROI trajectories are detrended, band-passed (`--band`, default
   6-40/min) and reduced with PCA; PC1 is the coherent body motion. The same
   computation on the static scene is the noise floor.
5. **Rate** - spectral peak, autocorrelation, and peak-to-peak intervals.

## Reading the result honestly

- **Resolution is set by clip length**: ±30/T breaths/min for a T-second shot.
  An 11 s shot cannot separate 15 from 20/min. Aim for 30-60 s.
- **Any slow motion lands in the band.** Posture shifts and head nods look
  like "breathing" over 3-4 cycles. Check the maps: real breathing shows on
  the shoulder line / torso, in phase, and not on the surroundings.
- **Disagreeing estimators mean irregular motion**; the tool says so.
- **Rate cannot establish sleep.** Normal adult rate is ~12-20/min both awake
  at rest and asleep. Sleep tends to be slightly slower and more *regular*,
  which needs minutes of signal to demonstrate, not seconds.

## Tests

```bash
uv run --with pytest pytest test_breathscope.py
```

Renders synthetic clips with a known oscillation under simulated pan + zoom
and checks the recovered rate and amplitude.
