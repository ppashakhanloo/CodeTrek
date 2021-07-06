import os

def gcp_copy_from(src, dst, bucket_name):
    os.system("gsutil -m cp -r " + "gs://" + bucket_name + "/" + src + " " + dst)

def gcp_copy_to(src, dst, bucket_name):
  os.system("gsutil -m cp -r " + src + " " + "gs://" + bucket_name + "/" + dst)
