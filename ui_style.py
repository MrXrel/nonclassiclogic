"""Presentation-only layout for the local scientific utility."""
import streamlit as st

CSS = """
<style>
:root {--utility-ink:#202124;--utility-muted:#666b70;--utility-rule:#d7d9dc;--utility-panel:#f2f3f4}
html,body,[data-testid="stApp"] {font-family:Arial,sans-serif;color:var(--utility-ink)}
[data-testid="stMainBlockContainer"] {max-width:none;padding:1.6rem 1.75rem 3rem}
h1 {font-family:Arial,sans-serif!important;font-size:24px!important;line-height:1.3!important;font-weight:600!important;letter-spacing:0!important;padding:0 0 .35rem!important}
h2 {font-size:19px!important;letter-spacing:0!important;font-weight:600!important}
h3 {font-size:16px!important;letter-spacing:0!important;font-weight:600!important;padding:.35rem 0!important}
p,label,[data-testid="stMarkdownContainer"] p {font-size:14px;line-height:1.5}
[data-testid="stCaptionContainer"] p {font-size:12px;color:var(--utility-muted);line-height:1.5}
[data-testid="stHeader"] {background:transparent;height:1.5rem}
[data-testid="stAppDeployButton"] {display:none}
[data-testid="stSidebar"] {background:var(--utility-panel);border-right:1px solid var(--utility-rule);width:176px!important;min-width:176px!important;max-width:176px!important}
[data-testid="stSidebarContent"] {padding-top:1rem}
[data-testid="stSidebarUserContent"] {padding:1rem .7rem}
.utility-brand {font-size:19px!important;font-weight:600;margin:0 0 1.6rem .45rem}
.st-key-section [role="radiogroup"] {gap:4px}
.st-key-section label {padding:9px 8px;border-left:3px solid transparent;margin:0!important;min-height:38px}
.st-key-section label[data-testid="stRadioOption"]>div>div:first-child {display:none}
.st-key-section label:has(input:checked) {border-left-color:#202124;background:#e7e8e9}
.st-key-section label:has(input:checked) p {font-weight:600}
.st-key-section label:focus-within {outline:2px solid #666;outline-offset:1px}
.st-key-section label p {font-size:14px;white-space:nowrap}
.utility-session {font-size:12px!important;color:#666b70;line-height:1.8!important}
[data-testid="stSidebar"] button {font-size:12px!important;padding:6px!important}
.st-key-workspace {gap:24px!important;align-items:flex-start!important}
.st-key-workspace>[data-testid="stLayoutWrapper"]:first-child {min-width:0;flex:1 1 0!important}
.st-key-workspace>[data-testid="stLayoutWrapper"]:last-child {width:280px!important;flex:0 0 280px!important;align-self:stretch}
.st-key-work_area {min-width:0;flex:1 1 0!important}
.st-key-settings_panel {width:280px!important;min-width:280px;flex:0 0 280px!important;border-left:1px solid var(--utility-rule);padding-left:22px;min-height:460px}
[data-testid="stVerticalBlock"] {gap:.8rem}
.st-key-settings_panel [data-testid="stVerticalBlock"] {gap:.7rem}
.st-key-settings_panel [data-testid="stForm"] {border:0;border-radius:0;padding:0}
[data-testid="stBaseButton-primary"] {background:#45494e!important;border:1px solid #45494e!important;color:#fff!important}
[data-testid="stBaseButton-primary"]:hover {background:#303438!important;border-color:#303438!important}
button,[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="textarea"] {border-radius:2px!important;box-shadow:none!important}
button {font-family:Arial,sans-serif!important;min-height:35px!important;font-size:14px!important}
[data-baseweb="input"],[data-baseweb="select"]>div {background:#fff;border-color:#bec2c6}
[data-testid="stNumberInput"] input {font-variant-numeric:tabular-nums}
[data-testid="stNumberInputContainer"],.stSelectbox .react-aria-ComboBox>[role="group"] {border:1px solid #bec2c6!important;border-radius:2px!important;background:#fff!important;min-height:38px}
[data-testid="stNumberInputField"],.stSelectbox input {background:#fff!important;color:#202124!important}
[data-testid="stExpander"] details {border:0;border-top:1px solid var(--utility-rule);border-radius:0!important}
[data-testid="stExpander"] summary {padding:10px 0;font-size:13px}
[data-testid="stAlert"] {border-radius:2px;padding:.6rem .8rem}
[data-testid="stMetricValue"] {font-size:18px!important}
[data-testid="stMetricLabel"] {font-size:12px!important}
[data-testid="stTabs"] [role="tablist"] {gap:18px;border-bottom:1px solid var(--utility-rule)}
[data-testid="stTabs"] [role="tab"] {font-size:13px;padding:8px 0}
.utility-values {width:100%;border-collapse:collapse;font-size:13px;margin:0 0 8px}
.utility-values th,.utility-values td {border:1px solid var(--utility-rule);padding:8px 10px;text-align:left;font-variant-numeric:tabular-nums}
.utility-values th {background:#f3f4f5;font-weight:600}
.utility-values td {background:#fff}
@media(min-width:768px) {
 [data-testid="stSidebarHeader"] {display:none}
 [data-testid="stSidebarContent"] {padding-top:0}
 [data-testid="stSidebarUserContent"] {padding:1.75rem .7rem;margin-top:0}
}
@media(min-width:1200px) {.st-key-settings_panel {position:sticky;top:24px}}
@media(max-width:1199px) {
 .st-key-workspace {flex-direction:column-reverse!important;gap:22px!important}
 .st-key-workspace>[data-testid="stLayoutWrapper"]:first-child,
 .st-key-workspace>[data-testid="stLayoutWrapper"]:last-child {width:100%!important;min-width:0;flex:0 0 auto!important}
 .st-key-settings_panel {width:100%!important;min-width:0;flex:0 0 auto!important;border-left:0;border-bottom:1px solid var(--utility-rule);padding:0 0 18px;min-height:0}
 .st-key-work_area {width:100%!important;flex:0 0 auto!important}
}
@media(max-width:640px) {
 [data-testid="stHeader"] {height:40px;background:#fff}
 [data-testid="stMainBlockContainer"] {padding:3.4rem .8rem 2rem}
 [data-testid="stTabs"] [role="tablist"] {gap:12px;overflow-x:auto}
 [data-testid="stTabs"] [role="tab"] {font-size:12px;white-space:nowrap}
 .utility-values th,.utility-values td {padding:6px;font-size:12px}
}
</style>
"""


def install_style():
    st.markdown(CSS, unsafe_allow_html=True)


def workspace():
    """Create both slots first, so controls can compute before results render."""
    with st.container(horizontal=True, wrap=False, key="workspace", gap="medium"):
        work = st.container(key="work_area", width="stretch")
        settings = st.container(key="settings_panel", width=280)
    return work, settings


def values_table(rows):
    """A small accessible, non-editable results table without a chart toolbar."""
    from html import escape
    rows = list(rows)
    if not rows:
        return
    names = list(rows[0])
    cells = lambda values, tag: "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
    body = "".join("<tr>" + cells([row.get(name, "") for name in names], "td") + "</tr>" for row in rows)
    st.markdown('<table class="utility-values"><thead><tr>' + cells(names, "th") +
                '</tr></thead><tbody>' + body + '</tbody></table>', unsafe_allow_html=True)
