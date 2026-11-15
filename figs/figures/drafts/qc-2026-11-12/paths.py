import json,sys,re
S='20250904_183921'
def walk(o,p,out):
    if isinstance(o,dict):
        for k,v in o.items():
            if S in str(k): out.add(re.sub(r'\[\d+\]','[]',p+'.'+'<SID>'))
            walk(v,p+'.'+str(k) if S not in str(k) else p+'.<SID>',out)
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v,p+'[]',out)
    elif isinstance(o,str) and S in o: out.add(p)
for f in sys.argv[1:]:
    o=json.load(open(f)); out=set(); walk(o,'',out)
    print('##',f); [print('   ',x) for x in sorted(out)[:12]]
