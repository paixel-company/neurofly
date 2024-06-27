# Neuronal Tools

Some tools for neuronal image analysis.

## Installation 

git clone git@github.com:beanli161514/neuronal-tools.git

cd neuronal-tools

pip install -r ./src/requirements.txt

pip install -e ./src/


## Making skeleton dataset

1. create dir with structure
-- labeled/
  -- skels/
  -- bg/

2. use simple_viewer to browse whole brain image, choose about 30 128^3 foreground image blocks and some bright background image blocks (vessels and other bright curvalinear structures)

3. use skel_annotator to label all the foreground image blocks.

4. augment foreground blocks,
    aug -source labeled/ -out labeled_aug/
