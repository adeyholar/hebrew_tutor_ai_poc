# patch_aeneas_hebrew.py - Modular Aeneas Patch for Unlisted Languages

import os
import site
import re

# Secure Path (find installed aeneas)
site_packages = site.getsitepackages()[0]
wrapper_path = os.path.join(site_packages, 'aeneas', 'ttswrappers', 'espeakngttswrapper.py')

if not os.path.exists(wrapper_path):
    raise FileNotFoundError(f"Aeneas wrapper not found at {wrapper_path}")

# Backup (idempotent, secure)
backup_path = wrapper_path + '.bak'
if not os.path.exists(backup_path):
    with open(wrapper_path, 'r', encoding='utf-8') as f:
        original = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original)
    print(f"Backup created: {backup_path}")

# Patch (add 'he' if not present)
with open(wrapper_path, 'r', encoding='utf-8') as f:
    code = f.read()

if "'he'" not in code:
    # Find SUPPORTED_LANGUAGES list, append 'he'
    pattern = r"SUPPORTED_LANGUAGES = \[(.*?)\]"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        lang_list = match.group(1)
        new_list = lang_list.rstrip() + ",\n        Language.HEBREW,\n    "
        new_code = code.replace(match.group(0), f"SUPPORTED_LANGUAGES = [{new_list}]")
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
        print(f"Patched {wrapper_path} to add Hebrew ('he') support.")
    else:
        print("SUPPORTED_LANGUAGES not found; manual edit required.")
else:
    print("Hebrew already supported; no patch needed.")