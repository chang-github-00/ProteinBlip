# ProteinBlip
MINIGPT4 IMPLEMENTATION FOR PROTEIN


## Build Environment
conda env create -f environment.yml
source activate minigpt4

## Pretraining Stage 1
```
bash run_scripts/run_pretraining_stage_1.sh
```

## Pretraining Stage 2
```
bash run_scripts/run_finetuning_stage_2.sh
```

## Evaluation with Simple Query
```
bash run_scripts/run_eval.sh
```

## Alignment Pre-training Data For Protein-Text


### Dataset Overview
Altogether 715k protein-text pair, including
- 146k data from PDB, using publication abstract as protein caption (align_data_split1.txt)
- 569k data from SwissProt, using SwissProt annotation as protein caption. (align_data_split2.txt)


### Data format
each line is a json entry, in the following format: 

```
{
    "id": pdb/ uniprot entry id,
    "source": entry source(e.g. swissprot),
    "caption": a text description for that protein,
    "sequence": AA sequence
}
    
```

### Dataset Curation 
- ```process_protchat.py``` can be used to curate align_data_split1.txt
- ```process_swissprot_caption.py``` can be used to curate align_data_split2.txt

### Need Further Checking
- Only include chain-A for PDB sequences
- SwissProt processing use hand-written rule for parsing and captioning, need additional checking, especially for "CC" and "FT"


## Protein Instruction Tuning Dataset


### Dataset Overview


| Task                                     | Source              | Entry-number | Protein-number | Style                     | Avg-len(prompt) | Avg-len(caption) | ID-Type    |
|------------------------------------------|---------------------|--------------|----------------|---------------------------|-----------------|------------------|------------|
| Secondary-Structure Prediction           | Netsurfp-2.0        | 9k           | 9k             | Free generation          | 16              | 84               | PDB        |
| Fold (structural Domain) Prediction      | SCOPe               | 62k          | 32k            | T/F + Free generation    | 33              | 21               | Scope id   |
| GO Prediction (bp, mf, cc)               | PDB + swissprot     | 2109k        | 433k           | T/F + Free generation    | 35              | 19 (T/F)         | PDB, UNIPROT |
| Enzyme Activity Prediction               | PDB + swissprot     | 232k         | 232k           | Free generation          | 23              | 23               | PDB, UNIPROT |
| Fitness Prediction (high/very high)      | ProteinGym          | 1474k        | 77 (other left for test) | Binned class + score     | 77              | 16               | UNIPROT    |
| Protein Design                          | Mol-Instruct        | 200k         |                  |                           |                 |                  |            |
| Protein-Protein Interaction              | BioGRID             | 3483k        | 65k            | T/F, multiple protein    | 17              | 1                | Partly has UNIPROT ID, in the format: Q12527:P53142 |
| Motif - (bind, dna-bind, intramembrane, transmembrane, Mutagen, peptide, signal peptide, zinc-finger, variants) | SwissProt | 1878k         | 338k           | Free generation          | 17              | 11               | UNIPROT    |
| Function Free-form Description            | MolInstruct         | 88k          | N/A            | Free generation          | 20              | 54               | N/A, note that id variable is not in json |
| All                                      |                     |              |                |                           |                 |                  |            |



### Data format
each line is a json entry, in the following format: 

For single protein tasks:
```
		{"prompt": "<protein><proteinHere></protein>",
		"sequence": "MEKEKKVKYFLRKSAFGLASVSAAFLVGST",
		"caption":"xxxxx",
		"id":"xxxx"}
    
```
For multiple protein tasks:
```
		{"prompt":"<protein><proteinHere_1></protein><protein><proteinHere_2></protein>…", 
		"sequence_1":"MEKEKKVKYFLRKSAFGLASVSAAFLVGST",
		"sequence_2":"MEKEKKVKYFLRKSAFGLASVSAAFLVGST",
		…
		"caption":"xxxxx",
		"id":"xxxx"}

```
