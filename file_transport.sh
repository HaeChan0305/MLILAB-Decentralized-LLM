scp root@69.30.85.133:/workspace/output_qwen_4_0/checkpoint-0/result.json root@69.30.85.27:~
# ssh root@69.30.85.133 -p 22191 -i ~/.ssh/id_ed25519
# ssh root@69.30.85.27 -p 22165 -i ~/.ssh/id_ed25519
scp -P 22191 -o "ProxyCommand=ssh -p 22165 root@69.30.85.27 -W %h:%p" root@69.30.85.133:/workspace/output_qwen_4_0/checkpoint-0/result.json root@69.30.85.27:~
