# sbb_column_classifier
This tool allows you to get the number of columns for a RGB document image.

Documents with more than 6 columns will be classified
as 6 column documents.

## Installation

      pip install .

## How to use

      sbb_column_classifier -i <image file name> -m <directory of models>

sbb_column_classifier can also produce JSON output with the `--json-out
example.json` option.

## Model
Models can be found here [qurator-data.de](https://qurator-data.de/eynollah/).
