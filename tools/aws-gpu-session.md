# One AWS GPU session: launch, verify, bring the data back, terminate

How the Hopper (p5.4xlarge), Blackwell (g7e.2xlarge) or L40S (g6e.xlarge) card is
rented and used to check what the simulator only knows from documentation. Written
for the agent or person who finally gets capacity; every step is a command.

Rules that apply (`~/.cache/pantheonsim-briefs/round3.md`, "AWS"): at most 2 GPU
instances alive across all sessions, 3 hours each, tag `owner=pantheonsim-nvidia`
and `purpose=<branch>`, write the ledger (`~/.cache/pantheonsim-aws/ledger.md`)
BEFORE working on an instance and when terminating it, never touch an instance you
did not launch, terminate the moment the data is copied and verify it. If the
permission system refuses any step, stop; do not route around it.

## 0. What the account allows

Quotas (us-east-1): Running On-Demand P instances 64 vCPUs (a p5.4xlarge has 16),
G and VT 64 (g7e.2xlarge 8, g6e.xlarge 4). Other regions: 0. Spot: 0. Capacity
Blocks: not offered for these types, and not enabled for the account. There is
no capacity API: the only test is a launch, and a launch that fails with
`InsufficientInstanceCapacity` costs nothing. On 2026-10-09 every zone of
us-east-1 refused all three types for the whole day (see the ledger); poll at
most four times an hour.

Reusable resources already in us-east-1 (nothing to create): key pair
`pantheon-bench-wsl` (the private key is `~/.ssh/id_rsa`), security group
`sg-0222be85a6de4185e` (`pantheon-bench-sg`, SSH from this machine's IP only; if the
IP changed, `AuthorizeSecurityGroupIngress` for `<ip>/32` and revoke the old rule
afterwards), the default VPC's per-zone subnets, and the Deep Learning Base OSS Nvidia
Driver GPU AMI (Ubuntu 24.04), newest `ami-028357be7d5b15c53` (20261006) at the time.

## 1. Try to launch (the AWS MCP tool, `mcp__aws-mcp__aws___run_script`)

```python
r='us-east-1'; ami='ami-028357be7d5b15c53'; sg='sg-0222be85a6de4185e'
subs={'us-east-1a':'subnet-09102d00163f47f60','us-east-1b':'subnet-0a5f8c6edeb0b80aa','us-east-1c':'subnet-0772e4662c22dd520',
      'us-east-1d':'subnet-07d275b496c32dd6f','us-east-1e':'subnet-017a08ef5c28ddc6f','us-east-1f':'subnet-03ecb5c862b8d3098'}
plan=[('g7e.2xlarge',['us-east-1b','us-east-1d']),('p5.4xlarge',list(subs)),('g6e.xlarge',['us-east-1a','us-east-1b','us-east-1c','us-east-1d'])]
got=None; log=[]
for t,azs in plan:
    for az in azs:
        try:
            d=await call_boto3(service_name='ec2',operation_name='RunInstances',region_name=r,params={
              'ImageId':ami,'InstanceType':t,'MinCount':1,'MaxCount':1,'KeyName':'pantheon-bench-wsl',
              'NetworkInterfaces':[{'DeviceIndex':0,'SubnetId':subs[az],'Groups':[sg],'AssociatePublicIpAddress':True}],
              'BlockDeviceMappings':[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':150,'VolumeType':'gp3','DeleteOnTermination':True}}],
              'MetadataOptions':{'HttpTokens':'required'},
              'TagSpecifications':[{'ResourceType':'instance','Tags':[{'Key':'Name','Value':'pantheonsim-r4-aws-verify'},
                  {'Key':'owner','Value':'pantheonsim-nvidia'},{'Key':'purpose','Value':'r4-aws-verify'}]}]})
            got=(t,az,d['Instances'][0]['InstanceId']); break
        except Exception as e:
            log.append((t,az,str(e)[:80]))
    if got: break
{'got':got,'log':log}
```

Only the first success matters; stop at one instance. Write the ledger line
(instance id, type, purpose, time, who, `LAUNCH`) at once. Then get the address:

```python
d=await call_boto3(service_name='ec2',operation_name='DescribeInstances',region_name='us-east-1',params={'InstanceIds':['<id>']})
i=d['Reservations'][0]['Instances'][0]; (i['State']['Name'], i.get('PublicIpAddress'))
```

If the instance runs but ssh never answers within ten minutes, terminate it (step 5)
and report.

## 2. Ship the tree and run the session (this machine)

```bash
cd ~/.cache/pantheonsim-r4-aws            # the branch with nvidia/tools/card-session-remote.sh
git archive --prefix=vgpu/ HEAD | gzip > /tmp/vgpu-tree.tgz
IP=<public ip>; SSH="ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR ubuntu@$IP"
scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /tmp/vgpu-tree.tgz ubuntu@$IP:/tmp/
$SSH 'tar xzf /tmp/vgpu-tree.tgz -C ~ && cd ~/vgpu && nohup nvidia/tools/card-session-remote.sh <slug> ~/out > ~/session.log 2>&1 &'
$SSH 'tail -5 ~/out/summary.txt'          # poll; the whole session is about 45-90 minutes
```

`<slug>` is the profile: `h100` (p5.4xlarge), `rtx-pro-6000` (g7e.2xlarge; the card is
the "Server Edition", whose profile is `rtx-pro-6000-server`: pass that name so the
expected files carry it, and add the `rtx-pro-6000` link if the simulator should also use
it), `l40s` (g6e.xlarge).

What it does (`nvidia/tools/card-session-remote.sh`; every step logs to `~/out/<step>.log`
and `~/out/summary.txt`):

1. `run_lowprec.sh lt|sparselt|cvt [ptx120 on sm_120] --card <slug> --update`: the transcripts
   `nvidia/tests/data/lowprec/<probe>.<slug>.txt` (cuBLAS 13.3 from a pip wheel when it
   installs, the L4 transcripts' version; cuSPARSELt and cuDNN from `nvidia-*-cu13` wheels).
2. `dnn_fp8_attention` and `dnn_fp8_attention_frontend` against NVIDIA's cuDNN.
3. The programs that were written from documentation, built for the card's own SASS and run
   natively (their expected last line is `PASS`): `dsmem_cluster cooperative_cluster setmaxnreg
   stmatrix vector_atomics wgmma_cute tma_*_cute` (sm_90a), `mma_blockscale narrow_cvt
   ldmatrix_forms` (sm_120a), `lt_paths lt_epilogue_paths lt_blockscaled_paths sparselt_paths`
   on the real cuBLASLt and cuSPARSELt. `tcgen05_gemm` needs sm_100a and is not run.
4. The characterization: `profile.yaml` (`characterize.cu` and the telemetry),
   `ptx_semantics` and `control_flow` reference outputs, `device_attributes --dump`, `nvidia-smi -q`,
   `ncu --query-metrics`.

## 3. Copy everything back

```bash
mkdir -p /tmp/aws-session-out && scp -r -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@$IP:'~/out' /tmp/aws-session-out/
scp -i ~/.ssh/id_rsa ... ubuntu@$IP:'~/vgpu/nvidia/tests/data/lowprec/*.<slug>.txt' nvidia/tests/data/lowprec/
```

Redact the unit's serial number, PDI and UUID from `nvidia-smi-q.txt` before committing it.
Commit the transcripts as expected files, then run the simulator against them:

```bash
nvidia/tests/e2e/run_lowprec.sh lt <slug>; nvidia/tests/e2e/run_lowprec.sh sparselt <slug>
nvidia/tests/e2e/run_lowprec.sh cvt <slug>; nvidia/tests/e2e/run_lowprec.sh ptx120 <slug>   # after a CI-style build
nvidia/tools/compare-profile.py /tmp/aws-session-out/out/profile.yaml nvidia/profiles/<slug>.yaml
```

Report the diffs and where the code is; flip `verified` only under the existing criteria
(the whole characterization read from the card, `nvidia/docs/profiles.md`).

## 4. Terminate the moment step 3 is done

```python
await call_boto3(service_name='ec2',operation_name='TerminateInstances',region_name='us-east-1',params={'InstanceIds':['<id>']})
# then poll DescribeInstances until State.Name == 'terminated', and write the ledger line TERMINATED
```

Nothing else was created (the key pair and security group are shared and stay).

## Notes from the 2026-10-09 session (g6e.2xlarge, L40S)

* Capacity: g7e.2xlarge, p5.4xlarge and g6e.xlarge were refused in every zone (default placement and zone by zone)
  at the polls of 17:46, 18:00 and 18:15 UTC; `g6e.2xlarge` (the same L40S, two more vCPUs) launched at the 18:30 poll in
  us-east-1b, and g7e.2xlarge, g7e.4xlarge and p5.4xlarge stayed refused at 18:45, 19:02 and 19:17. Try the neighbouring
  sizes (g6e.2xlarge, g7e.4xlarge) in every poll. A launch call can take over
  100 seconds to fail (three retries inside the SDK): wrap it in `asyncio.wait_for`, launch one type at a time, and if the
  tool times out, list instances tagged `purpose=<branch>` before doing anything else.
* The instance (8 vCPUs, 61 GB, the DLAMI with CUDA 12.8, 12.9, 13.0 and 13.2) builds the simulator in about 15 minutes
  (`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8 -- -k`), which lets the simulator's side of
  every comparison run next to the card: `VGPU_BUILD_DIR=$HOME/vgpu/build PATH=/usr/local/cuda-13.2/bin:$PATH
  nvidia/tests/e2e/run_lowprec.sh lt|sparselt|cvt` and `nvidia/tools/verify-profile.sh nvidia/<slug> <dir with
  <slug>.ptx_semantics.ref.txt and <slug>.control_flow.ref.txt>`. Build with CUDA 13.2 (the shims need its cuBLASLt
  emulation types; 13.0's headers fail in `cublaslt_api.cpp`); that tree needs the `runtime_api.cpp` fix for
  `cudaMemcpyNodeParams::reserved` (an int in 13.2, int[3] before). `vgpucupti` and `vgpunvenc` do not build on the
  AMI (no `cupti.h` beside the CUDA 13.2 include path, no `cuda_runtime.h` for the NVENC target); `-k` skips them.
* The whole remote session took 3 minutes on the L40S (the probes are fast; the 1278-descriptor cuBLASLt sweep took 30
  seconds). The cost of a session is the wait for capacity, not the run.

### Second half of the same night (g7e.2xlarge, RTX PRO 6000)

* us-east-1 never gave a g7e.2xlarge or p5.4xlarge (13 polls from 17:46 to 22:42 UTC). Quota increases through the Service
  Quotas API (`RequestServiceQuotaIncrease`, G and VT 8 vCPUs and P 16 vCPUs in us-west-2 and us-east-2) were
  auto-approved for G within about an hour (P stays `CASE_OPENED`), and a g7e.2xlarge launched in us-east-2a at 22:42 on the fifth
  poll that included that region (the AMI there is `ami-089d4ee74147726da`). The region needed its own key pair and
  security group (`ImportKeyPair` with `~/.ssh/id_rsa.pub`, `CreateSecurityGroup` in the default VPC with SSH from this
  machine's /32); both were deleted afterwards. us-west-2 already has `pantheon-bench-wsl` and `pantheon-bench-sg`
  (`sg-0bcd2db068be2fc66`, AMI `ami-0fec8acfc603e6e16`).
* The slug for this card is `rtx-pro-6000-server` (the profile that exists), not `rtx-pro-6000`.
* The remote session ran in 2.5 minutes; building the simulator on the instance took 8 minutes on 8 vCPUs.
  Everything was copied back in the first 15 minutes, and the instance was terminated after 17 minutes of use.
  Do the card-only extras (PROBE_DUMP of `lt` and `sparselt`, the native `sparselt_paths` with the wheel's
  `LD_LIBRARY_PATH`) before terminating: nothing can be re-asked of the card afterwards.
