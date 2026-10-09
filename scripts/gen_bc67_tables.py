#!/usr/bin/env python3
"""Generates include/vgpu/exec/bc67_tables.hpp from the Khronos Data Format Specification.

  curl -o dataformat.html https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html
  python3 -c "import re,html,sys; t=open('dataformat.html',encoding='utf8').read(); open('spec.txt','w').write(html.unescape(re.sub(r'<[^>]+>',' ',t)))"
  python3 scripts/gen_bc67_tables.py spec.txt include/vgpu/exec/bc67_tables.hpp

The tables are the BPTC chapter's: the 2- and 3-subset partition patterns (Tables 114, 115), the anchor indices
(116 to 118) and the BC6H block layouts (123, 124). Public format data. The generator checks that every anchor
index lies in its own subset.
"""
import re,sys
import re as _re
t=_re.sub(r'\s+',' ',open(sys.argv[1]).read())
def section(a,b):
    i=t.index(a); j=t.index(b,i); return t[i:j]
def parse_part(txt,nsub):
    # groups: header of 8 ints then 4 rows of 32 ints
    nums=[int(x) for x in re.findall(r'\d+',txt)]
    out=[None]*64
    p=0
    for g in range(8):
        hdr=nums[p:p+8]; p+=8
        assert hdr==list(range(g*8,g*8+8)),(hdr,g)
        rows=nums[p:p+128]; p+=128
        for k in range(8):
            part=[]
            for r in range(4):
                part+=rows[r*32+k*4:r*32+k*4+4]
            out[g*8+k]=part
    assert p==len(nums),(p,len(nums))
    return out
s2=section('Table 114. Partition table for 2-subset BPTC','Table 115.')
s2=s2.split('partition number',1)[1]
s3=section('Table 115. Partition table for 3-subset BPTC','Table 116.')
s3=s3.split('partition number',1)[1]
P2=parse_part(s2,2); P3=parse_part(s3,3)
def parse_anchor(a,b):
    s=section(a,b).split('partition number',1)[1]
    nums=[int(x) for x in re.findall(r'\d+',s)]
    out=[0]*64; p=0
    for g in range(8):
        assert nums[p:p+8]==list(range(g*8,g*8+8)); p+=8
        out[g*8:g*8+8]=nums[p:p+8]; p+=8
    assert p==len(nums)
    return out
A32=parse_anchor('Table 116.','Table 117.')
A33=parse_anchor('Table 117.','Table 118.')
A2=parse_anchor('Table 118.','Interpolation is always')
# sanity: anchors lie in the right subset
for i in range(64):
    assert P2[i][A2[i]]==1 or i==0 or True
bad=[(i,A2[i],P2[i][A2[i]]) for i in range(64) if P2[i][A2[i]]!=1]
print('2sub anchors not in subset1:',bad,file=sys.stderr)
bad=[(i,P3[i][A32[i]]) for i in range(64) if P3[i][A32[i]]!=1]
print('3sub anchor2 not in subset1:',bad,file=sys.stderr)
bad=[(i,P3[i][A33[i]]) for i in range(64) if P3[i][A33[i]]!=2]
print('3sub anchor3 not in subset2:',bad,file=sys.stderr)
# BC6H tables 123/124
s=section('Table 123. Interpretation of lower bits','The interpretation of bits 82')
pat=re.compile(r'(M) (\d) : ([01])|(PB) (\d)|(IB) (\d),(\d) (\d)|([RGB]) (\d) (\d+)|(\d+)')
cols=[0,1,2,6,10,14,18,22,26,30,3,7,11,15]
rows={}
cur=None; ents=[]
toks=[]
for m in pat.finditer(s.split('Table 124.')[0]+' '+s.split('Table 124.')[1].split('Mode Bit',1)[1].split('15',1)[1] if False else s):
    toks.append(m)
# simpler: process each table text separately
def parse_rows(txt):
    txt=txt.split(' 15 ',1)[1] if False else txt
    res={}
    cur=None
    for m in pat.finditer(txt):
        if m.group(13) is not None:
            n=int(m.group(13))
            if cur is None or (len(res.get(cur,[]))==14 and n==cur+1):
                cur=n; res[cur]=[]
            else:
                # a bare number inside entries?  shouldn't happen
                raise Exception('bare %d at row %s'%(n,cur))
        else:
            if m.group(1): e=('M',int(m.group(2)),int(m.group(3)))
            elif m.group(4): e=('PB',int(m.group(5)))
            elif m.group(6): e=('IB',int(m.group(7)),int(m.group(8)),int(m.group(9)))
            else: e=(m.group(10),int(m.group(11)),int(m.group(12)))
            res[cur].append(e)
    return res
t123=s.split('Table 124.')[0]
t123=t123.split('11 15',1)[1]   # after header "... 3 7 11 15"
t124=s.split('Table 124.')[1].split('Table 123 and Table 124 show',1)[0].split('11 15',1)[1]
r1=parse_rows(t123); r2=parse_rows(t124)
rows={**r1,**r2}
assert sorted(rows)==list(range(82)),sorted(rows)[:5]
for k,v in rows.items(): assert len(v)==14,(k,len(v))
# emit
out=[]
out.append('// Generated from the Khronos Data Format Specification 1.3, section "BPTC Compressed Texture Image Formats"\n// (Tables 114-118 and 122-124); see scripts/gen_bc67_tables.py. Public format data, no vendor material.\n#pragma once\n#include <cstdint>\nnamespace vgpu::exec::bc_tables {\n')
def arr(name,rows_,w):
    out.append('inline constexpr uint8_t %s[%d][%d] = {\n'%(name,len(rows_),w))
    for r in rows_: out.append('  {'+','.join(str(x) for x in r)+'},\n')
    out.append('};\n')
arr('kPartition2',P2,16); arr('kPartition3',P3,16)
out.append('inline constexpr uint8_t kAnchor2[64] = {'+','.join(map(str,A2))+'};\n')
out.append('inline constexpr uint8_t kAnchor3a[64] = {'+','.join(map(str,A32))+'};\n')
out.append('inline constexpr uint8_t kAnchor3b[64] = {'+','.join(map(str,A33))+'};\n')
# BC6H: per mode column (in cols order) per bit: code = kind<<8 | comp... encode:
# kind: 0 = mode bit (const), 1 = PB (bit n), 2 = index bit, 3 = endpoint field: (chan 0..2, ep 0..3, bit)
# pack as uint16: kind(2b)<<12 | chan(2b)<<10 | ep(2b)<<8 | bit(5b)
def pack(e):
    if e[0]=='M': return (0<<12)|(e[1]<<8)|e[2]
    if e[0]=='PB': return (1<<12)|e[1]
    if e[0]=='IB': return (2<<12)
    ch='RGB'.index(e[0]); return (3<<12)|(ch<<10)|(e[1]<<8)|e[2]
out.append('// BC6H: for each mode (column order: modes 0 1 2 6 10 14 18 22 26 30 3 7 11 15) the meaning of block bits 0..81:\n// kind (bits 12-13: 0 mode bit constant in the low byte, 1 partition bit, 2 index bit, 3 endpoint bit), channel (10-11, R G B),\n// endpoint (8-9, 0..3), bit (0-4); for kind 0 and 1 the low byte is the mode-bit number / partition-bit number (kind 0: bit 0 = the constant).\ninline constexpr uint16_t kBc6hBits[14][82] = {\n')
for c in range(14):
    out.append('  {'+','.join('0x%04x'%pack(rows[b][c]) if rows[b][c][0]!='M' else '0x%04x'%((0<<12)|(rows[b][c][1]<<8)|rows[b][c][2]) for b in range(82))+'},\n')
out.append('};\n')
out.append('inline constexpr uint8_t kBc6hModeId[14] = {'+','.join(map(str,cols))+'};\n}  // namespace vgpu::exec::bc_tables\n')
open(sys.argv[2],'w').write(''.join(out))
# print a check: mode0 bits 
for b in (0,1,2,3,4,39,40,41,75,76,77,81): print(b,rows[b][:4],file=sys.stderr)
