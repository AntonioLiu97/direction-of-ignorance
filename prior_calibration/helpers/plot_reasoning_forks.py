"""Editable forked-path presentation of three measured divergence examples."""
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'plot_export'
FORK_STYLE = dict(width=10, font=10, title_font=13, line_inches=.175,
                  red='#c83232', blue='#336b8d', gray='#92979d', dpi=220,
                  question_wrap=113, prefix_wrap=110, path_wrap=55,
                  show_title=False, show_footer=False)
CASES = [('cookies',2115,'Birthday cookies: include the host'),
         ('manuscripts',461,'Manuscript copies: distinguish one copy from ten'),
         ('course',6666,'Course duration: distinguish one week from 24')]

def wrapped(text, width):
    return '\n'.join(textwrap.fill(p,width=width,break_long_words=False,
                     break_on_hyphens=False) if p else '' for p in text.split('\n'))

def render_case(name, index, title, style=None):
    s = dict(FORK_STYLE, **(style or {}))
    p = next(x for x in json.loads((ROOT/'fig4_divergence_1536_v1/pairs.json').read_text())['pairs'] if x['train_index']==index)
    r = json.loads((ROOT/f'fig4_{name}_logits_v1/result.json').read_text())
    assert r['both_argmax_match'], 'Replay argmax mismatch: do not present as reproduced.'
    assert r['question']==p['question'] and r['prompt_sha256']==p['prompt_sha256']
    frequency = json.loads((ROOT/'fig4_divergence_1536_v1/candidate_frequencies.json').read_text())
    counts = frequency['counts']
    # One scale for the union of displayed candidates: token colors stay fixed
    # across branches even when their top-five membership or rank changes.
    candidate_ids = {t['token_id'] for arm in r['arms'].values() for t in arm['top7'][:5]}
    values = [counts[str(i)] for i in candidate_ids]
    assert min(values)>0
    norm = LogNorm(vmin=min(values),vmax=max(values))
    cmap = LinearSegmentedColormap.from_list('token_commonness',[s['blue'],'#eee8e4',s['red']])
    k = p['first_divergence_zero_based']
    cuts = {a:p[a]['offsets'][k][0] for a in ['baseline','tuned']}
    prefix = p['baseline']['response'][:cuts['baseline']]
    assert prefix == p['tuned']['response'][:cuts['tuned']]
    question = wrapped(p['question'],s['question_wrap'])
    shared = wrapped(prefix+' ____________',s['prefix_wrap']) if prefix else '____________  (divergence at the first generated token)'
    paths = {a:wrapped(p[a]['response'][cuts[a]:],s['path_wrap']) for a in cuts}
    qh = .45 + (question.count('\n')+1)*s['line_inches']
    sh = .4 + (shared.count('\n')+1)*s['line_inches']
    th = .4 + max(t.count('\n')+1 for t in paths.values())*s['line_inches']
    fork_h, chart_h = .9, 2.15
    top_h = .55 if s.get('show_title', False) else .25
    footer_h = .55 if s.get('show_footer', False) else .2
    height = top_h + qh + sh + fork_h + chart_h + .55 + th + footer_h
    fig = plt.figure(figsize=(s['width'],height),facecolor='white')
    def txt(x,top,text,**kw):
        return fig.text(x,1-top/height,text,va='top',fontsize=s['font'],
                        color='black',parse_math=False,**kw)
    if s.get('show_title', False):
        fig.text(.05,1-.2/height,title,va='top',fontsize=s['title_font'],weight='bold')
    y = top_h
    txt(.05,y,'Question',weight='bold');txt(.05,y+.25,question)
    y+=qh
    txt(.05,y,'Shared generation',weight='bold');txt(.05,y+.25,shared)
    y+=sh
    txt(.5,y,'forking paths',ha='center',weight='bold')
    for center in [.26,.68]:
        fig.add_artist(FancyArrowPatch((.5,1-(y+.23)/height),(center,1-(y+.61)/height),
            transform=fig.transFigure,arrowstyle='-|>',mutation_scale=12,color='black',linewidth=1.2))
    y+=fork_h
    chart_bottom = 1-(y+chart_h)/height
    chart_height = chart_h/height-.05/height
    axes = []
    for a,x,label in [('baseline',.08,'α = 1'),('tuned',.48,'α = 0.65')]:
        txt(x+.16,y-.18,label,ha='center',weight='bold')
        ax = fig.add_axes([x,chart_bottom,.32,chart_height])
        axes.append(ax)
        top = r['arms'][a]['top7'][:5]
        vals = [v['probability']*100 for v in top]
        colors = [cmap(norm(counts[str(v['token_id'])])) for v in top]
        ax.barh(np.arange(5),vals,color=colors,height=.62)
        ax.set_yticks(np.arange(5));ax.set_yticklabels([repr(v['token']) for v in top],fontsize=9,fontfamily='monospace')
        ax.invert_yaxis()
        maxv=max(t['probability']*100 for arm in r['arms'].values() for t in arm['top7'][:5])
        ax.set_xlim(0,maxv*1.23)
        for i,v in enumerate(vals):ax.text(v+maxv*.015,i,f'{v:.2f}%',va='center',fontsize=9)
        ax.set_xlabel('Next-token probability (%)',fontsize=9)
        ax.tick_params(axis='x',labelsize=9)
        ax.spines[['top','right']].set_visible(False)
    cax = fig.add_axes([.84, chart_bottom, .02, chart_height])
    cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='vertical')
    cb.set_ticks([min(values),max(values)])
    cb.set_ticklabels(['Less common','More common'])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=8,length=0)
    cb.set_label('Empirical Pile frequency (log scale)',fontsize=8)
    y+=chart_h+.55
    for a,x in [('baseline',.05),('tuned',.53)]:
        txt(x,y,'Divergent continuation',weight='bold')
        fig.text(x,1-(y+.27)/height,paths[a],va='top',fontsize=s['font'],
                 linespacing=1.25,color=s['red'] if a=='baseline' else s['blue'],parse_math=False)
    if s.get('show_footer', False):
        foot=('Color scale shared across the displayed candidates in both panels; '
              'bar length is next-token probability.\n'
              'Top 5 per panel; unit-temperature softmax; greedy decoding. '
              'Both replay choices match saved generations.')
        fig.text(.05,.15/height,foot,va='bottom',fontsize=8,color='#555')
    OUT.mkdir(exist_ok=True)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'fig4_{name}_fork.{ext}',dpi=s['dpi'],facecolor='white',
                    bbox_inches='tight',pad_inches=.15)
    return fig

def render_cases(cases=CASES, style=None, output_name='fig4_reasoning_forks'):
    """Render all examples on a US-letter page with per-example color scaling."""
    s = dict(FORK_STYLE)
    s.update(
        width=8.5, font=9, title_font=10,
        question_wrap=118, prefix_wrap=118, path_wrap=62,
        # Body weight 500 (medium) reads heavier than DejaVu's 400 without
        # collapsing the contrast with the bold (700) headers.
        family='Inter', weight=500, mono_family='Source Code Pro',
        margin=.3, column_gap=.255, label_width=.64, xlim_pad=1.23, gap_scale=1,
        prompt_label='Question', fork_label='Greedy continuation at α = {}',
        colorbar=True, side_bar_width=.55,  # colorbar: True, False, or 'vertical'
    )
    s.update(style or {})
    body_font = dict(fontfamily=s['family'], weight=s['weight'])
    mono_font = dict(fontfamily=s['mono_family'], weight=s['weight'])
    pairs = json.loads((ROOT/'fig4_divergence_1536_v1/pairs.json').read_text())['pairs']
    counts = json.loads(
        (ROOT/'fig4_divergence_1536_v1/candidate_frequencies.json').read_text()
    )['counts']

    examples = []
    for name, index, title in cases:
        p = next(x for x in pairs if x['train_index'] == index)
        r = json.loads((ROOT/f'fig4_{name}_logits_v1/result.json').read_text())
        assert r['both_argmax_match'], 'Replay argmax mismatch: do not present as reproduced.'
        assert r['question'] == p['question'] and r['prompt_sha256'] == p['prompt_sha256']

        k = p['first_divergence_zero_based']
        cuts = {a: p[a]['offsets'][k][0] for a in ['baseline', 'tuned']}
        prefix = p['baseline']['response'][:cuts['baseline']]
        assert prefix == p['tuned']['response'][:cuts['tuned']]
        candidate_ids = {
            t['token_id'] for arm in r['arms'].values() for t in arm['top7'][:5]
        }
        values = [counts[str(i)] for i in candidate_ids]
        assert min(values) > 0
        examples.append(dict(
            name=name,
            title=title,
            question_raw=p['question'],
            shared_raw=prefix+' ____________' if prefix
                       else '____________  (divergence at the first generated token)',
            paths_raw={a: p[a]['response'][cuts[a]:] for a in cuts},
            result=r,
            frequency_norm=LogNorm(vmin=min(values), vmax=max(values)),
        ))

    cmap = LinearSegmentedColormap.from_list(
        'token_commonness', [s['blue'], '#eee8e4', s['red']]
    )
    max_probability = max(
        t['probability'] * 100
        for example in examples
        for arm in example['result']['arms'].values()
        for t in arm['top7'][:5]
    )

    width, height = s['width'], 11
    # Vertical rhythm (inches), tuned for a 9 pt body; `gap_scale` shrinks it
    # for smaller fonts. Text heights are measured from the rendered artists
    # rather than estimated, so these gaps are the real whitespace.
    g = s['gap_scale']
    gaps = dict(
        top=.26*g, title_to_question=.06*g, question_to_shared=.14*g,
        shared_header_to_text=.04*g, shared_to_columns=.16*g,
        column_header_to_chart=.05*g, chart_to_text=.1*g,
        text_to_divider=.14*g, divider_to_title=.16*g, chart_h=.5*g,
    )
    # Inches reserved below the content for the color bar (or a bottom pad).
    colorbar_band = (.72 if s['colorbar'] is True else .08) * g
    content_limit = height - colorbar_band

    def build(font, height):
        # Measure at the export dpi: at the default 100 dpi, text extents are
        # quantized so coarsely that 8.5 pt and 9 pt measure identically.
        fig = plt.figure(figsize=(width, height), dpi=s['dpi'], facecolor='white')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        wrap_scale = s['font'] / font

        def inches(artist):
            return artist.get_window_extent(renderer).height / fig.dpi

        def txt(x, top, text, **kwargs):
            fontsize = kwargs.pop('fontsize', font)
            return fig.text(
                x, 1-top/height, text, va='top', fontsize=fontsize,
                color='black', parse_math=False, **{**body_font, **kwargs}
            )

        def fitted(x, top, raw, wrap, max_width, **kwargs):
            """Wrap `raw` so its rendered width fits in `max_width` inches.

            Character counts only approximate width, so start generous and
            tighten the wrap until the measured extent fits the column.
            """
            while True:
                artist = txt(x, top, wrapped(raw, wrap), **kwargs)
                too_wide = artist.get_window_extent(renderer).width / fig.dpi > max_width
                if not too_wide or wrap <= 20:
                    return artist
                artist.remove()
                wrap -= 1

        # Horizontal layout: thin outer margins, two equal columns. Spacing is
        # fixed in inches (tuned at 8.5 in) so narrower figures keep room for
        # the token labels and percentage annotations.
        margin, column_gap = s['margin']/width, s['column_gap']/width
        full_width = (1 - 2*margin) * width
        column_width = (1 - 2*margin - column_gap) / 2
        columns = {'baseline': margin, 'tuned': margin + column_width + column_gap}
        column_inches = column_width * width
        label_width = s['label_width']/width  # token labels left of each bar axis
        # A vertical color bar sits right of the charts; both charts give up the
        # same width so their bar lengths stay comparable.
        side_bar = s['side_bar_width']/width if s['colorbar'] == 'vertical' else 0

        y = gaps['top']
        for example_index, example in enumerate(examples):
            label = (s['prompt_label'] if len(examples) == 1
                     else f'{s["prompt_label"]} {chr(ord("A") + example_index)}')
            title = txt(margin, y, label,
                        fontsize=font + 1.5, weight='bold')
            y += inches(title) + gaps['title_to_question']

            question = fitted(margin, y, example['question_raw'],
                              round(s['question_wrap'] * wrap_scale * 1.3), full_width)
            y += inches(question) + gaps['question_to_shared']

            header = txt(margin, y, 'Shared generation', weight='bold')
            y += inches(header) + gaps['shared_header_to_text']
            shared = fitted(margin, y, example['shared_raw'],
                            round(s['prefix_wrap'] * wrap_scale * 1.3), full_width)
            y += inches(shared) + gaps['shared_to_columns']

            headers = [
                txt(columns['baseline'] + column_width/2, y,
                    s['fork_label'].format('1'), ha='center', weight='bold'),
                txt(columns['tuned'] + column_width/2, y,
                    s['fork_label'].format('0.65'), ha='center', weight='bold'),
            ]
            y += max(inches(h) for h in headers) + gaps['column_header_to_chart']

            chart_bottom = 1-(y+gaps['chart_h'])/height
            axes_bottom_inches = 0
            for arm, column_x in columns.items():
                ax = fig.add_axes([column_x + label_width, chart_bottom,
                                   column_width - label_width - .01 - side_bar,
                                   gaps['chart_h']/height])
                top = example['result']['arms'][arm]['top7'][:5]
                probabilities = [token['probability']*100 for token in top]
                colors = [
                    cmap(example['frequency_norm'](counts[str(token['token_id'])]))
                    for token in top
                ]
                ax.barh(np.arange(5), probabilities, color=colors, height=1)
                ax.set_yticks(np.arange(5))
                ax.set_yticklabels(
                    [repr(token['token']) for token in top],
                    fontsize=font-1.5, **mono_font
                )
                ax.invert_yaxis()
                ax.set_xlim(0, max_probability*s['xlim_pad'])
                ax.set_ylim(4.5, -.5)
                for row, probability in enumerate(probabilities):
                    ax.text(
                        probability+max_probability*.015, row,
                        f'{probability:.2f}%', va='center', fontsize=font-1.5,
                        **body_font
                    )
                ax.tick_params(axis='x', labelsize=font-1.5, pad=2)
                for label in ax.get_xticklabels():
                    label.set(**body_font)
                ax.spines[['top', 'right']].set_visible(False)
                bbox = ax.get_tightbbox(renderer)
                axes_bottom_inches = max(
                    axes_bottom_inches, height - bbox.y0 / fig.dpi)
            if side_bar:
                cax = fig.add_axes([1 - margin - side_bar + .02, chart_bottom,
                                    .08/width, gaps['chart_h']/height])
                colorbar = fig.colorbar(
                    ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
                    cax=cax, orientation='vertical'
                )
                colorbar.set_ticks([0, 1])
                colorbar.set_ticklabels(['Less\ncommon', 'More\ncommon'])
                colorbar.ax.minorticks_off()
                colorbar.ax.tick_params(labelsize=font-1.5, length=0, pad=2)
                for label in colorbar.ax.get_yticklabels():
                    label.set(linespacing=1, **body_font)
                colorbar.ax.get_yticklabels()[0].set_va('bottom')
                colorbar.ax.get_yticklabels()[1].set_va('top')
            y = axes_bottom_inches + gaps['chart_to_text']

            paths = [
                fitted(column_x, y, example['paths_raw'][arm],
                       round(s['path_wrap'] * wrap_scale * 1.3), column_inches,
                       linespacing=1.3)
                for arm, column_x in columns.items()
            ]
            y += max(inches(p) for p in paths)

            if example_index < len(examples)-1:
                y += gaps['text_to_divider']
                fig.add_artist(plt.Line2D(
                    [margin, 1-margin], [1-y/height]*2,
                    transform=fig.transFigure, color='#c8c8c8', linewidth=.8
                ))
                y += gaps['divider_to_title']
        return fig, y

    # Pick the largest font that fits a US-letter page, then trim the page to
    # the content so no empty band is left above the color bar.
    for font in np.arange(s['font'], s['font']-2.01, -.25):
        fig, content_bottom = build(float(font), height)
        plt.close(fig)
        print(f'  {font}pt body: content needs {content_bottom:.2f} in '
              f'(limit {content_limit:.2f})')
        if content_bottom <= content_limit:
            break
    else:
        raise RuntimeError(
            f'Content needs {content_bottom:.2f} in; cannot fit US letter.')
    height = content_bottom + colorbar_band
    fig, content_bottom = build(font, height)
    print(f'Fork figure body font {font}pt; page {width} x {height:.2f} in.')

    if s['colorbar'] is True:
        cax = fig.add_axes([.25, .3*g/height, .5, .12*g/height])
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
            cax=cax, orientation='horizontal'
        )
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(['Less common', 'More common'])
        colorbar.ax.minorticks_off()
        colorbar.ax.tick_params(labelsize=font-1.5, length=0, pad=2)
        for label in colorbar.ax.get_xticklabels():
            label.set(**body_font)

    OUT.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(
            OUT/f'{output_name}.{ext}', dpi=s['dpi'], facecolor='white',
        )
    return fig

if __name__=='__main__':
    fig = render_cases()
    plt.close(fig)
