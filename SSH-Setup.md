# Configuring CSE234 Image

Here's a step-by-step, tailored to your situation. You're already signed in as `emc001` on a non‑CSE234 `vscode-dsmlp` pod, so the plan is: (1) close this session cleanly, (2) configure SSH on your **local** machine so the proxy command launches the CSE 234 pod, (3) reconnect from VS Code/Cursor, (4) start RapidFire AI inside the new pod.

Everything in steps 1–6 is run on your **local** machine (macOS/Linux terminal or Windows PowerShell). Steps 7+ are inside the new pod.

---

### 1. Close your current Cursor/VS Code remote session

In this window: `File → Close Remote Connection` (or just quit Cursor). That stops VS Code from holding the current pod open, but the pod itself may keep running until it times out — we'll clean it up in step 2 if needed.

### 2. (Optional, recommended) Delete the existing non‑CSE234 pod

From your **local** terminal:

```bash
ssh emc001@dsmlp-login.ucsd.edu
kubectl get pod
kubectl delete pod vscode-dsmlp   # use the actual name from `get pod`
exit
```

DSMLP only lets you run one interactive pod at a time per user, so deleting the current one first avoids "pod already exists" errors when the new launch script runs.

### 3. Make sure you have an SSH key locally

On your **local** machine:

```bash
ls ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub 2>/dev/null
```

If nothing is there, create one:

```bash
ssh-keygen -t ed25519 -C "emc001@ucsd"
```

Accept the default path (`~/.ssh/id_ed25519`). The `.pub` file is the one you share; the one with no extension is private — never copy that anywhere.

### 4. Push your public key to the login node

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub emc001@dsmlp-login.ucsd.edu
```

You'll enter your **SSO password** once and approve a **Duo push**. Verify it worked:

```bash
ssh emc001@dsmlp-login.ucsd.edu
```

It should log in **without** asking for your password (Duo push will still be required once every ~8 hours).

### 5. Add a `cse234` host to your local SSH config

Edit `~/.ssh/config` on your **local** machine (create it if it doesn't exist) and append:

```sshconfig
Host cse234
  User emc001
  ForwardAgent yes
  IdentityFile ~/.ssh/id_ed25519
  ProxyCommand ssh -i ~/.ssh/id_ed25519 emc001@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch-cse234.sh -W CSE234_SP26_A00 -H -N vscode-dsmlp -c 8 -m 32
```

A few notes on that `ProxyCommand`:

- `launch-cse234.sh` is the CSE 234–specific launcher (this is what gives you the `cse234` conda env you saw missing in your current pod).
- `-W CSE234_SP26_A00` is the course workspace (Spring 2026).
- `-H` keeps the pod headless for SSH attach.
- `-N vscode-dsmlp` names the pod `vscode-dsmlp` (what `kubectl delete pod` will target later).
- `-c 8 -m 32` = 8 CPUs, 32 GB RAM. Tune down if you just want to smoke‑test.

Permissions matter — SSH will silently refuse the config/key otherwise:

```bash
chmod 600 ~/.ssh/config ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

### 6. Connect from Cursor/VS Code using the new host

In Cursor: `Cmd/Ctrl+Shift+P → Remote‑SSH: Connect to Host… → cse234`.

- First connect takes a while: the `ProxyCommand` is SSHing to the login node and running `launch-cse234.sh`, which schedules a Kubernetes pod with the CSE 234 image and attaches.
- If it asks about host key fingerprint, type `yes`.
- If it hangs for >2 min, in a local terminal run `ssh -v cse234` to watch what the proxy command is doing (usually Duo or a `pod already exists` error).

### 7. Inside the new pod: initialize conda + activate `cse234`

Once VS Code's remote window opens, open a terminal (Ctrl+`) and:

```bash
conda init bash
exec bash   # or just close and reopen the terminal
conda activate cse234
```

Confirm you're on the right image:

```bash
conda env list          # should show `cse234` in the list
which python            # should be under /opt/conda/envs/cse234/...
python -c "import rapidfireai; print(rapidfireai.__version__)"
```

If `conda activate cse234` still fails, you're not on the CSE 234 image — recheck step 5's `ProxyCommand` (wrong workspace or wrong launcher script).

### 8. Launch RapidFire AI

Still in the pod terminal, from your project directory (e.g. `~/rapidfireAI-rag-context-engineering`):

```bash
rapidfireai init --evals
rapidfireai start
```

You can skip `init --evals` if `~/rapidfireai/rf_mode.txt` already contains just the single line `evals`.

### 9. Forward ports so you can open the panel

In VS Code/Cursor, open the **PORTS** tab (bottom panel next to Terminal) and forward at least:

- `8851` (usually auto‑forwarded)
- `8853` — this is the RapidFire AI panel
- Any other port RapidFire prints in the start logs

Then visit `http://127.0.0.1:8853` in your browser.

### 10. When you're done — stop the pod

Don't just close Cursor; the pod keeps running. From your **local** terminal:

```bash
ssh emc001@dsmlp-login.ucsd.edu
kubectl get pod                # confirm pod name
kubectl delete pod vscode-dsmlp
```

Make sure any unsaved work is committed/pushed first (your home dir is persistent, but the pod's `/tmp`, `/scratch`, etc. is not guaranteed).

Also monitor your cost here [Google Sheet.](https://docs.google.com/spreadsheets/d/1RDjqEO1KE4MKbEsYbaXfrnXoiv4yGwcFaMi3NtS7QjA/edit?gid=1747936408#gid=1747936408)

---

### Quick Troubleshooting Table


| Symptom                                   | Likely cause                                        | Fix                                                                    |
| ----------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| `ssh cse234` prompts for SSO password     | `ssh-copy-id` didn't land, or wrong `IdentityFile`  | Redo step 4; check `~/.ssh/authorized_keys` on login node              |
| Proxy command fails: "pod already exists" | Old `vscode-dsmlp` pod still running                | Step 2: `kubectl delete pod vscode-dsmlp`                              |
| No `cse234` env after connect             | Launched the wrong image (what happened to you now) | Verify `ProxyCommand` uses `launch-cse234.sh` and `-W CSE234_SP26_A00` |
| `rapidfireai: command not found`          | `cse234` env not activated                          | `conda activate cse234` (and confirm `which rapidfireai`)              |
| Port 8853 not reachable                   | Not forwarded                                       | Add it manually in the PORTS tab                                       |
| Duo pushes every reconnect                | Normal; token lasts ~8h                             | Just approve                                                           |


One thing worth flagging from the doc itself: the note at the top warns that the RapidFire AI interactive controller panel is not yet reliable over VS Code port forwarding even when 8853 is forwarded early. If you hit that, the panel UI may render but controller actions hang — that's a known issue, not your SSH setup.