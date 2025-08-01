#!/usr/bin/env python3
"""
Button Conflict Finder - Find all duplicate button keys in Streamlit code
"""

import re
from collections import defaultdict

def find_button_conflicts():
    """Find all button conflicts in the UI file"""
    
    try:
        with open('unified_uel_ui.py', 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("❌ unified_uel_ui.py not found!")
        return
    
    # Find all button calls
    button_patterns = [
        r'st\.button\("([^"]*)"([^)]*)\)',  # st.button calls
        r'st\.form_submit_button\("([^"]*)"([^)]*)\)'  # form submit buttons
    ]
    
    all_buttons = []
    key_usage = defaultdict(list)
    buttons_without_keys = []
    
    line_number = 1
    for line in content.split('\n'):
        for pattern in button_patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                button_text = match.group(1)
                params = match.group(2)
                
                # Extract key if present
                key_match = re.search(r'key="([^"]*)"', params)
                if key_match:
                    key_name = key_match.group(1)
                    key_usage[key_name].append((line_number, button_text))
                else:
                    buttons_without_keys.append((line_number, button_text, line.strip()))
                
                all_buttons.append((line_number, button_text, params))
        
        line_number += 1
    
    print("🔍 BUTTON CONFLICT ANALYSIS")
    print("=" * 50)
    print(f"📊 Total buttons found: {len(all_buttons)}")
    print(f"🔑 Buttons with keys: {len(all_buttons) - len(buttons_without_keys)}")
    print(f"❌ Buttons without keys: {len(buttons_without_keys)}")
    print()
    
    # Show duplicate keys
    duplicates_found = False
    print("🚨 DUPLICATE KEYS FOUND:")
    print("-" * 30)
    for key, usage_list in key_usage.items():
        if len(usage_list) > 1:
            duplicates_found = True
            print(f"❌ Key '{key}' used {len(usage_list)} times:")
            for line_num, button_text in usage_list:
                print(f"   Line {line_num}: \"{button_text}\"")
            print()
    
    if not duplicates_found:
        print("✅ No duplicate keys found!")
    print()
    
    # Show buttons without keys
    if buttons_without_keys:
        print("⚠️  BUTTONS WITHOUT KEYS:")
        print("-" * 30)
        for line_num, button_text, full_line in buttons_without_keys[:10]:  # Show first 10
            print(f"Line {line_num}: \"{button_text}\"")
            print(f"    {full_line}")
            print()
        
        if len(buttons_without_keys) > 10:
            print(f"... and {len(buttons_without_keys) - 10} more")
    else:
        print("✅ All buttons have keys!")
    
    return duplicates_found, buttons_without_keys

def auto_fix_buttons():
    """Automatically fix all button conflicts"""
    
    print("\n🔧 AUTO-FIXING BUTTON CONFLICTS...")
    
    with open('unified_uel_ui.py', 'r') as file:
        content = file.read()
    
    lines = content.split('\n')
    fixed_lines = []
    button_counter = 3000  # Start high to avoid conflicts
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        
        # Find st.button calls without keys
        button_pattern = r'(st\.button\("([^"]*)"([^)]*)\))'
        
        def fix_button(match):
            nonlocal button_counter
            full_match = match.group(1)
            button_text = match.group(2)
            params = match.group(3)
            
            # Check if key already exists
            if 'key=' in params:
                return full_match  # Already has key
            
            # Generate unique key
            key_base = re.sub(r'[^a-zA-Z0-9]', '_', button_text.lower())
            key_base = key_base[:20]  # Limit length
            unique_key = f"{key_base}_{button_counter}"
            button_counter += 1
            
            # Add key parameter
            if params.strip() and not params.strip().startswith(','):
                new_button = f'st.button("{button_text}", {params.strip()}, key="{unique_key}")'
            else:
                new_button = f'st.button("{button_text}"{params}, key="{unique_key}")'
            
            print(f"   Fixed line {line_num}: \"{button_text}\" → key=\"{unique_key}\"")
            return new_button
        
        # Apply fixes
        fixed_line = re.sub(button_pattern, fix_button, line)
        
        # Also fix form_submit_button
        form_button_pattern = r'(st\.form_submit_button\("([^"]*)"([^)]*)\))'
        
        def fix_form_button(match):
            nonlocal button_counter
            full_match = match.group(1)
            button_text = match.group(2)
            params = match.group(3)
            
            # Form submit buttons don't need keys (they're in forms)
            return full_match
        
        fixed_line = re.sub(form_button_pattern, fix_form_button, fixed_line)
        fixed_lines.append(fixed_line)
    
    # Write back
    with open('unified_uel_ui.py', 'w') as file:
        file.write('\n'.join(fixed_lines))
    
    print("✅ Auto-fix completed!")

if __name__ == "__main__":
    print("🔍 Analyzing button conflicts in unified_uel_ui.py...")
    
    duplicates, no_keys = find_button_conflicts()
    
    if duplicates or no_keys:
        print("\n🔧 Would you like to auto-fix these issues? (y/n): ", end="")
        response = input().lower()
        
        if response == 'y':
            auto_fix_buttons()
            print("\n✅ All button conflicts should now be fixed!")
            print("🚀 Try running your app: python main.py --run")
        else:
            print("❌ Please fix the conflicts manually before running the app.")
    else:
        print("✅ No button conflicts found! Your app should work fine.")