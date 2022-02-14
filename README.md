# sbb_column_classifier
This tool allows you to get the number of columns for a RGB document image.

Documents with more than 6 columns will be classified
as 6 column documents.

## Installation

      pip install .

## How to use

      sbb_column_classifier -i <image file name> -m <directory of models>

sbb_column_classifier can also produce a SQLite3 database file with results,
using the `--db-out results.sqlite3` option.

## Model
Models can be found here [qurator-data.de](https://qurator-data.de/eynollah/).
