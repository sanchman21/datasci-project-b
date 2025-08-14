import os
import argparse
import glob
import shutil

def find_replace_in_files(directory="./mlruns", find_string="///workspace/CMML-v2/townim", replace_string=".", file_extensions=["*.yaml", "*.yml"]):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    files = []
    
    for ext in file_extensions:
        files.extend(glob.glob(os.path.join(directory, f"**/{ext}"), recursive=True))
        
    files = list(set(files))
    
    if not files:
        print(f"No YAML files found in '{directory}' or its subdirectories.")
        return
    
    print(f"Found {len(files)} YAML file(s) to process.")
    
    for file_path in sorted(files):
        print(f"\nProcessing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if find_string not in content:
                print(f"  No occurrences of '{find_string}' found.")
                continue
            
            backup_path = file_path + '.bak'
            shutil.copy2(file_path, backup_path)
            print(f"  Created backup: {backup_path}")
            new_content = content.replace(find_string, replace_string)
            changes_made = content != new_content
            
            if changes_made:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                print(f"  Replaced '{find_string}' with '{replace_string}'.")
            else:
                print(f"  No changes made (identical content).")
                
            old_lines = content.splitlines()
            new_lines = new_content.splitlines()
            
            for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
                if old_line != new_line:
                    print(f"  Line {i+1}:")
                    print(f"    Old: {old_line}")
                    print(f"    New: {new_line}")
                    
        except Exception as e:
            print(f"  Error processing '{file_path}': {str(e)}")
            
def main():
    find_replace_in_files()
    print("\nFind-and-replace operation completed.")
    
if __name__ == "__main__":
    main()