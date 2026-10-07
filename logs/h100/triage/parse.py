"""parse.py <junit.xml> -> one line per test: nodeid | outcome | first error line"""
import sys, re, xml.etree.ElementTree as ET
for tc in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    cls, name = tc.get("classname"), tc.get("name")
    nid = cls.replace(".", "/") + ".py::" + name
    f = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
    if f is None:
        out, msg = ("skipped" if tc.find("skipped") is not None else "passed"), ""
    else:
        out = "failed"
        txt = f.text or f.get("message", "")
        lines = [l for l in txt.splitlines() if l.startswith("E ") and l[1:].strip()]
        msg = lines[0][1:].strip() if lines else f.get("message", "")[:200]
        # prefer the root-cause line of dynamo errors
        for l in lines:
            if "Explanation:" in l or "got AttributeError" in l: msg += " || " + l[1:].strip(); break
    print(f"{nid}\t{out}\t{msg[:400]}")
