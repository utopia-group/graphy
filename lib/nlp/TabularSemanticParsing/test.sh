dataset=movies

# ./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --demo 0 --demo_db $dataset --checkpoint_path model/bridge-spider-bert-large-ems-70-1-exe-68-2.tar
./experiment-bridge.sh configs/bridge/spider-bridge-bert-large.sh --demo 0 --demo_db $dataset --checkpoint_path model/test/model-best.tar