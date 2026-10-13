# GeoFuse troubleshooting

---

## `No module named geofuse`

The package wasn't linked correctly during setup.

**Fix:** Re-run the installer (`install.bat` / `install.sh`) and confirm a success message at the end. If it still fails, manually run inside the activated environment:

```bash
pip install -e .
```

---

## `Conda Not Found`

The installer script fails immediately.

**Fix:**
* Confirm Miniforge/Anaconda is installed.
* Ensure the Conda `Scripts/` (Windows) or `bin/` (Linux/macOS) directory is on your `PATH`.
* If installed in a custom location, paste the correct path when prompted by the script.

---

## `[WinError 127]` or DLL Load Failure on Windows

A GDAL or GDAL-dependent library (GeoPandas, rasterio) fails to import.

**Cause:** `torch` / `geofuse` DLLs must be loaded before the GDAL stack on Windows.

**Fix:** Never import `geopandas`, `rasterio`, or similar libraries before `import torch` or `from geofuse import ...` in custom scripts. The entry points (`ui/app.py` and the `geofuse` command) already enforce the correct order.

---

## `Windows fatal exception: code 0xc06d007f` from `numpy.linalg.cholesky` (or `MixedLM`)

A fatal exception terminates the interpreter with no Python traceback the moment a matrix operation runs. Most often surfaces when running the mixed-effects fusion (`statsmodels.regression.mixed_linear_model.MixedLM.fit`).

**Cause:** Conda-forge's MKL 2026.x ships an INTEL OpenMP threading runtime whose symbols clash with `numpy.linalg.cholesky` and matrix-multiply (`gemm`) on Windows. The mismatch causes the OS loader to fail entry-point resolution and kill the process.

**Fix:** Switch the env's BLAS metapackage from MKL to openblas, then reinstall `numpy` and `scipy` via conda so they bind to the new BLAS:

```bash
conda activate geofuse
conda install -c conda-forge "libblas=*=*openblas" "libcblas=*=*openblas" "liblapack=*=*openblas" -y
pip uninstall numpy scipy -y
conda install -c conda-forge numpy scipy --force-reinstall -y
```

Fresh installs run by the current `install.bat` / `install.sh` already pin openblas + conda-installed numpy/scipy (see `scripts/setup_env.py`), so this fix is only needed for envs built manually, or before this recent fix.

---

## `DecompressionBombWarning`

A warning appears in the terminal when downloading very high-resolution Street View images.

**This is safe to ignore.** GeoFuse automatically handles large image limits and resizes panoramas before segmentation.

---

## `Earth Engine authentication error` / `ee.Initialize` fails

Google Earth Engine credentials are missing or expired.

**Fix:**
1. Run `geofuse auth earthengine` in the activated environment and follow the browser prompt.
2. Without a browser (a cluster login node), connect with `ssh -L 8085:localhost:8085 HOST` and open the printed address on your own computer, or use `geofuse auth earthengine --no-browser` and paste the code it shows.
3. For a service account, `geofuse credentials set earthengine-service-account KEY.json`.
4. "No project found": `geofuse config set earthengine.project YOUR-PROJECT`.

A background job never waits on a sign-in page where no browser can open; it stops with the command to run.

---

## GVI stops with "Google's Street View service no longer answers as the GVI downloader expects"

Google changed one of the endpoints the scraper uses. The job stops instead of reporting every point as having no panorama. `geofuse doctor` runs one search and shows the same message. The downloader has to be updated to the endpoint Google now uses. Meanwhile, a Street View API key and `download_mode = "api"` download through Google's documented API instead.

---

## GVI stops with "Google refused the Street View API request"

Google checked the key and refused it. The message carries Google's reason:

* **API key not valid:** the key was mistyped or deleted. Store it again with `geofuse credentials set streetview-key`.
* **Map Tiles API has not been used in project … or it is disabled:** enable the **Map Tiles API** for the key's Google Cloud project, wait a few minutes, and run again.
* **Billing / permission errors:** turn billing on for the project, or lift API or address restrictions on the key that exclude this machine.

`geofuse doctor` runs one API search and shows the same reason. A run in API mode with no key stored stops before it starts (`download_mode = "api" needs a Street View API key`).

---

## `geofuse doctor` warns that the caches are on a network filesystem

The panorama cache and the NDVI tile index are SQLite files, which are unsafe on NFS / SMB / Lustre / GPFS when several machines write at once. Move them to a local or scratch disk:

```bash
geofuse config set paths.caches /scratch/$USER/geofuse-caches
```

---

## "No usable keyring" when storing a credential

Headless Linux machines often have no keyring service. With `[credentials] store = "auto"` (the default) GeoFuse falls back to `credentials.toml` in your configuration folder, readable by you only; `geofuse config set credentials.store file` makes that explicit.

---

## `geofuse watch` says it needs `textual`

Install it into the environment (`pip install textual`), or follow jobs with `geofuse status` and `geofuse logs JOB -f`.

---

## The Jobs tab or `geofuse watch` says a remote's link is down

The remote's jobs are shown as they were when last read, and requests are not sent. Run `geofuse remote attach NAME` in a terminal and keep it open; it asks for the login there.

* **`the remote side closed the link (exit code 127)`:** the remote shell cannot find `geofuse`. Set the full start command, e.g. `geofuse remote add NAME HOST --command "conda run -n geofuse geofuse"` (or `module load ...; geofuse`).
* **`exit code 255`:** `ssh` itself failed: host name, user, key or network. Try `ssh USER@HOST` by hand.
* **It keeps reconnecting:** the link retries with growing pauses, up to a minute apart; it recovers by itself once the connection does.

---

## Out-of-Memory (GPU) Errors

The segmentation model runs out of VRAM during batch processing.

**Fix:**
* Set **GPU Batch Size** (`batch_size`) back to 1.
* The recorded CUDA graph holds its own memory. When recording fails the log says `CUDA graph capture failed` and the run continues layer by layer, slower but with the same values.
* Process fewer study areas simultaneously.
* The engine automatically falls back from CUDA → MPS → CPU if a device runs out of memory mid-job.

---

## GVI is slow, or its `Throughput:` line reports throttle events

The line at the end of each GVI run shows where the time went.

* **Throttle events, slow downloads:** Google is limiting requests from your address. The run backs off by itself; lower **Concurrent Downloads** (`download_concurrency`) so it trips less often. A larger pool does not lift Google's limit.
* **No throttle events, GPU busy well below 90 %:** downloads are the bottleneck. Raise `download_concurrency`.
* **GPU busy near 100 %:** the GPU is the limit. Check that the line says `fp16` (the default); a faster card, or several shards on several GPUs (`shards = N` in a run file), is the way up from there.

---

## `RuntimeError: main thread is not in main loop` when browsing for a file

Tk requires the thread that owns its interpreter to be the process's main
thread. Streamlit runs the script body, and every `on_click` callback, on a
per-rerun ScriptRunner thread, so a `tkinter` dialog opened in-process fails as
soon as any Tk state outlives the thread that created it.

The dialog therefore runs in a **child process** (`ui/file_picker_child.py`),
launched per click and gone before the next rerun. If you still see this error,
the app is running an older copy of `ui/file_picker.py`. Restart Streamlit so
the module is re-imported.

Related: if the picker reports *"The native file dialog could not open"* and
offers a path box instead, that is not this bug. The dialog opens on the machine
running the Streamlit **server**, so it cannot appear when the server is remote
and the browser is elsewhere. Paste the server-side absolute path into the box.

---

## GVI Job Shows as "Interrupted" After a Streamlit Restart

The job was running when Streamlit (or the machine) restarted. The job state was saved but the worker process was killed.

**Fix:** Switch to the tab the job came from (GVI or NDVI). Re-upload the **same** study area file you originally used. A restart panel appears above the input form with **Resume Job** and **Discard** buttons. **Resume Job** continues from the exact point where the job stopped; already-processed points and cached panoramas are skipped.

If the file you re-upload does not match the original geometry, the restart panel will refuse to resume. Discard the interrupted job and start a new run instead.

---

## Where Are Per-Job Logs Stored?

Every job writes a persistent text log to `logs/jobs/<job_id>.log`. The in-UI expander only keeps the last 100 lines in memory; the file on disk has the full log indefinitely.

**Quick access:** open the job's expander in the sidebar Job Monitor and click "📄 Open log file", which opens it in your OS default text editor.

External library logs (Google Earth Engine, Optuna) are routed into the same per-job file so they no longer appear in the host terminal.

---

## GeoTIFF File Is Huge or Mostly Empty (GVI)

This happens when a single bbox-wide raster is rendered for a study area whose points are clustered in a few small regions (e.g. neighbourhoods across multiple cities). The bbox contains millions of empty cells.

**Fix:** Switch to **GeoPackage** as the GVI output format; it is the recommended default for sparse or national-scale data. If you still want raster output, enable **Save GeoTIFF** in the GVI tab and GeoFuse will write one dense GeoTIFF per spatial cluster into `[Filename]_gvi_tiles/` along with a `tiles_index.json` describing each tile's bounds. No empty cells.
