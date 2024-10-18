import os
import sys
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import logging

logging.basicConfig(filename='run_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

batches = [1, 8, 64]
seqlen_lists = [1024, 2048, 8192, 16384]
# batches = [8]
# seqlen_lists = [8192]
configurations = [
    {'model': 'mamba2-1.3b', 'nheads': 64, 'ngroups': 1, 'headdim': 64, 'dstate': 128, 'chunk_size': 256},
    {'model': 'mamba2-2.7b', 'nheads': 80, 'ngroups': 1, 'headdim': 64, 'dstate': 128, 'chunk_size': 256},
]
# batches = [8]
# seqlen_lists = [8192]
# configurations = [
#     {'model': 'mamba2-2.7b', 'nheads': 80, 'ngroups': 1, 'headdim': 64, 'dstate': 128, 'chunk_size': 256},
# ]


# # Profile torch implementation for chunk_scan
# logging.info("Running Torch implementation of chunk_scan")
# from torch_chunk_scan import run_torch

# for batch in batches:
#     for seqlen in seqlen_lists:
#         for config in configurations:
#             try: 
#                 tflops, avg_latency = run_torch(batch, 
#                     config['nheads'], 
#                     config['ngroups'], 
#                     seqlen, 
#                     config['headdim'], 
#                     config['dstate'], 
#                     config['chunk_size'])
                
#                 logging.info(f"Running model: {config['model']} with batch={batch}, seqlen={seqlen}")
#                 logging.info(f"TFLOPS: {tflops:.2f}")
#                 logging.info(f"Avg Latency: {avg_latency:.2f} ms")

#             except Exception as e:
#                 logging.error(f"Failed to run model: {config['model']} with batch={batch}, seqlen={seqlen}")
#                 logging.error(f"Error: {e}")
#             logging.info('-' * 100)

# # Profile TensorRT implementation for chunk_scan
# logging.info("Running TensorRT implementation of chunk_scan")
# from trt_chunk_scan import run_trt

# for batch in batches:
#     for seqlen in seqlen_lists:
#         for config in configurations:
#             try: 
#                 tflops, avg_latency = run_trt(batch, 
#                     config['nheads'], 
#                     config['ngroups'], 
#                     seqlen, 
#                     config['headdim'], 
#                     config['dstate'], 
#                     config['chunk_size'])
                
#                 logging.info(f"Running model: {config['model']} with batch={batch}, seqlen={seqlen}")
#                 logging.info(f"TFLOPS: {tflops:.2f}")
#                 logging.info(f"Avg Latency: {avg_latency:.2f} ms")

#             except Exception as e:
#                 logging.error(f"Failed to run model: {config['model']} with batch={batch}, seqlen={seqlen}")
#                 logging.error(f"Error: {e}")
#             logging.info('-' * 100)

# Profile torch implementation for chunk_state
logging.info("Running Torch implementation of chunk_state")
from torch_chunk_state import run_torch

for batch in batches:
    for seqlen in seqlen_lists:
        for config in configurations:
            try: 
                tflops, avg_latency = run_torch(batch, 
                    config['nheads'], 
                    config['ngroups'], 
                    seqlen, 
                    config['headdim'], 
                    config['dstate'], 
                    config['chunk_size'])
                
                logging.info(f"Running model: {config['model']} with batch={batch}, seqlen={seqlen}")
                logging.info(f"TFLOPS: {tflops:.2f}")
                logging.info(f"Avg Latency: {avg_latency:.2f} ms")

            except Exception as e:
                logging.error(f"Failed to run model: {config['model']} with batch={batch}, seqlen={seqlen}")
                logging.error(f"Error: {e}")
            logging.info('-' * 100)