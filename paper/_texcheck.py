import re
from collections import Counter

tex = open('main.tex', encoding='utf-8').read()
bib = open('references.bib', encoding='utf-8').read()
bs = chr(92)

env_name = r'(\w+\*?)'
brace_open = re.escape('{')
brace_close = re.escape('}')

begins = re.findall(re.escape(bs + 'begin') + brace_open + env_name + brace_close, tex)
ends = re.findall(re.escape(bs + 'end') + brace_open + env_name + brace_close, tex)
cb, ce = Counter(begins), Counter(ends)
print("environments balanced:", cb == ce, "" if cb == ce else (cb - ce, ce - cb))
print("document env begin/end:", begins.count('document'), ends.count('document'))

cited = set()
for m in re.findall(re.escape(bs + 'cite') + brace_open + r'([^}]*)' + brace_close, tex):
    for k in m.split(','):
        cited.add(k.strip())
bibkeys = set(re.findall(r'@\w+' + brace_open + r'([^,]+),', bib))
missing = cited - bibkeys
print("cite keys:", len(cited), " bib entries:", len(bibkeys))
print("missing bib keys:", sorted(missing) if missing else "NONE")
print("unused bib entries:", sorted(bibkeys - cited))

t = re.sub(re.escape(bs) + r'[{}]', '', tex)
print("curly brace balance (open-close):", t.count('{') - t.count('}'))
print("TODO-verify markers in bib:", bib.count('TODO-verify'))
print("dataneeded uses in tex:", tex.count('dataneeded'))
thm = re.findall(re.escape(bs + 'begin') + brace_open + r'(theorem|lemma|proposition|corollary)' + brace_close, tex)
print("theorem-like count:", len(thm))
