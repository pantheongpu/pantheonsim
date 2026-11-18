import re,glob,os
def samples(f):
    out=[]
    for l in open(f):
        if l.startswith("host_gpu_sample:"):
            g=dict(re.findall(r"(\w+)=(\S+)",l)); out.append(g)
    return out
tot=0;diff=0
for f in sorted(glob.glob(os.path.expanduser("~/r11-on/*/*.txt"))):
    p=os.path.basename(f)[:-4]; off=os.path.expanduser("~/r11-off/")+f.split("/")[-2]+"/"+os.path.basename(f)
    if not os.path.exists(off): continue
    a=samples(off); b=samples(f); tot+=1
    ka={(g["kernel"],g["blocks"]) for g in a}
    new=[g for g in b if (g["kernel"],g["blocks"]) not in ka]
    gone=[g for g in a if (g["kernel"],g["blocks"]) not in {(g["kernel"],g["blocks"]) for g in b}]
    if new or gone or len(a)!=len(b):
        diff+=1
        print(p,"off",len(a),"on",len(b))
        for g in new: print("   new:",g["kernel"][:34],g["blocks"],"x",g["threads_per_block"],"est",g["estimate_s"],"send",g["would_send"],"cost",g["cost_s"],"launch_s",g.get("launch_s"))
        for g in gone: print("   gone:",g["kernel"][:34],g["blocks"])
print(tot,"programs compared;",diff,"differ")
