import  gc

def GCWrapper(func):
  def wrapper(*args, **kwargs):
    gc.collect()
    ret = func(*args, **kwargs)
    gc.collect()
    return ret
  return wrapper
