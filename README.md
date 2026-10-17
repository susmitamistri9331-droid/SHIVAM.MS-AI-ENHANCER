# SHIVAM.MS AI ENHANCE V7 — persistence and QUNQE recovery update

## Install / run in Colab
1. Select **Runtime → Change runtime type → T4 GPU** (if available).
2. Run `SHIVAM_MS_AI_ENHANCE_V7_COLAB_CODE.py` as one cell.
3. Upload your password-protected `SHIVAM_MS_AI_ENHANCE_V7...zip`, confirm, and launch.
4. Approve Google Drive access. Persistent data is stored under `MyDrive/SHIVAM_MS_AI_ENHANCE/` with `jobs`, `accounts`, `models`, `processed`, and per-account folders.

## Account and queue storage
- Account database is stored in the persistent account storage directory; passwords use salted PBKDF2 hashes.
- Each account has `QUNQE Processes/Original Videos`, `QUNQE Processes/Leftover Videos`, `QUNQE Processes/Resume Checkpoints`, and `Processed Videos`.
- QUNQE selection uses the source video path, not an ID shown to the user. Internal checkpoint keys remain private to keep different jobs isolated.
- Process names rename the saved original and recovery copy. The recovery filename is updated to `NAME__LEFT_Npct.mp4` as frame progress advances.
- The dropdown lists latest saved videos from both Original Videos and Leftover Videos. Selecting one resumes its matching saved process from the contiguous checkpoint frames.
- A completed video is renamed to the process name and saved in Processed Videos.

## Important persistence note
Google Drive files survive a Colab runtime reset, but a running GPU job does not keep running after Colab disconnects. Reopen the notebook, launch the same app ZIP, sign in to the same app account, refresh the QUNQE queue, and resume the saved process.


## Resume-path fix
- The saved Original and LEFTOVER recovery video paths no longer change on every progress update.
- Queue refresh preserves the currently selected resume/source path.
- The saved-video resume button resolves the selected path back to its QUNQE job and resumes from saved frame checkpoints.
- Queue diagnostics now show percent complete, percent left, stage, and any captured error.
