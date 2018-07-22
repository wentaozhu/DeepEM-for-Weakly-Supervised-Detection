config = {'test_data_path':['/mnt/easy/wentaobaidu/153/tianchi/newtest/'],
          'val_data_path':['/mnt/easy/wentaobaidu/153/tianchi/newtest/'], 
          'train_data_path':['/mnt/easy/wentaobaidu/153/tianchi/newtest/'], 
          
          'train_preprocess_result_path':'/mnt/easy/wentaobaidu/153/tianchi/preprocessing/newtest/',
          'val_preprocess_result_path':'/mnt/easy/wentaobaidu/153/tianchi/preprocessing/newtest/',  
          'test_preprocess_result_path':'/mnt/easy/wentaobaidu/153/tianchi/preprocessing/newtest/', 
          
          'train_annos_path':'/mnt/easy/wentaobaidu/153/tianchi/luna16/CSVFILES/annotations.csv',
          'val_annos_path':'/mnt/easy/wentaobaidu/153/tianchi/csv/newtest/annotations.csv',
          'test_annos_path':'/mnt/easy/wentaobaidu/153/tianchi/csv/newtest/annotations.csv',

          'black_list':['LKDS-00192', 'LKDS-00319', 'LKDS-00238', 'LKDS-00926', 'LKDS-00504',
                        'LKDS-00648', 'LKDS-00829', 'LKDS-00931', 'LKDS-00359', 'LKDS-00379', 
                        'LKDS-00541', 'LKDS-00353', 'LKDS-00598', 'LKDS-00684', 'LKDS-00065'],
          
          'preprocessing_backend':'python',

          'luna_segment':'/mnt/easy/wentaobaidu/153/tianchi/luna16/seg-lungs-LUNA16/',
          'preprocess_result_path':'/mnt/easy/wentaobaidu/153/tianchi/luna16/preprocess/',
          'luna_data':'/mnt/easy/wentaobaidu/153/tianchi/luna16/',
          'luna_label':'/mnt/easy/wentaobaidu/153/tianchi/luna16/CSVFILES/annotations.csv'
         } # LKDS-00648 - end is found by the labelmapping function
# 'LKDS-00192','LKDS-00319','LKDS-00238','LKDS-00926', 'LKDS-00504' is found from preprocessing
# 'LKDS-00504',
