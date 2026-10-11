import json

p = 'notebooks/final_project.ipynb'
nb = json.load(open(p, encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code' and 'auto-generated findings' in ''.join(c['source']):
        src = (
 "# auto-generated findings text (values pulled from the actual comparison tables above)\n"
 "from IPython.display import Markdown, display\n"
 "g = cmp.drop_duplicates(['task', 'model']).set_index(['task', 'model'])\n"
 "gg = cmp_g.drop_duplicates(['task', 'model']).set_index(['task', 'model'])\n"
 "rows_find = ['\n"
 "for task in ['category', 'price_tier', 'sentiment']:\n"
 "    b = float(g.loc[(task, 'logreg_test'), 'f1_macro'])\n"
 "    m = float(g.loc[(task, 'mtl_shared_encoder'), 'f1_macro'])\n"
 "    m_g = float(gg.loc[(task, 'mtl_shared_encoder'), 'f1_macro'])\n"
 "    verdict = 'MTL wins' if m > b else ('baseline wins' if m < b else 'tie')\n""    rows_find.append('- **{}**: logreg F1_macro={:.4f}, MTL F1_macro={:.4f} ({}); grouped-category re-run MTL F1_macro={:.4f}'.format(task, b, m, verdict, m_g))\n"
        "display(Markdown(chr(10).join(rows_find)))\n"
)
        c['source'] = [s + '\n' for s in src.splitlines(keepends=True) if s]
        print('rewrote findings cell', i)

json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'c{i}', 'exec')
print('all code cells compile OK')
