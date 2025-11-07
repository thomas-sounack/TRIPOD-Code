from main_extractor import clone_and_extract_tree
url = "http://qdiabetes.org/"
clone_dir = "C:/Users/clone_dir"
output_dir = "C:/Users/output_dir"

tree, concatenated_text = clone_and_extract_tree(url, clone_dir, output_dir)