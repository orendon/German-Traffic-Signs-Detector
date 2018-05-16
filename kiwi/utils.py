import sys

def download_progress(blocknum, blocksize, totalsize):
  """ Display download progress bar in the command line.
      Reference: https://stackoverflow.com/questions/13881092/download-progressbar-for-python-3 """

  readsofar = blocknum * blocksize
  if totalsize > 0:
    percent = readsofar * 1e2 / totalsize
    s = "\r%5.1f%% %*d / %d" % (
      percent, len(str(totalsize)), readsofar, totalsize)
    sys.stderr.write(s)
    if readsofar >= totalsize: # near the end
      sys.stderr.write("\n")
  else: # total size is unknown
    sys.stderr.write("read %d\n" % (readsofar,))