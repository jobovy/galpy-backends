"""frames.py <junit.xml>: per failed test, the deepest galpy frame and the key error detail"""
import sys, re, xml.etree.ElementTree as ET
for tc in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    f = tc.find("failure")
    if f is None: f = tc.find("error")
    if f is None: continue
    txt = f.text or ""
    name = tc.get("name")
    # dynamo "from user code:" block lists the user frames, deepest last
    uc = txt.split("from user code:")
    if len(uc) > 1:
        fr = re.findall(r'File "([^"]*galpy-fb/galpy/[^"]+)", line (\d+), in (\w+)\n\s*(.+)', uc[-1])
    else:
        fr = re.findall(r'^(galpy/[^:\n]+):(\d+): in (\w+)', txt, re.M)
        fr = [(a, b, c, "") for a, b, c in fr]
        if not fr:
            fr = re.findall(r'(/scratch/[^:\n]*galpy-fb/galpy/[^:\n]+):(\d+): in (\w+)', txt)
            fr = [(a, b, c, "") for a, b, c in fr]
    got = re.search(r"got (\w+Error\([^\n]{0,160})", txt)
    deep = fr[-1] if fr else ("?", "?", "?", "")
    print(f"{name}\n    {deep[0].split('galpy-fb/')[-1]}:{deep[1]} in {deep[2]}  {deep[3].strip()[:110]}")
    if got: print(f"    {got.group(1)}")
