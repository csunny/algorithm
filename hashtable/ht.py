#!/usr/bin/env python3

from hashlib import md5

class HT:
    """
    This is a simple hash table impletment.
    """
    def __init__(self):
        self.ht_table = dict()
    
    def _cus_hash(self, key):
        """
        自定义的hash函数
        """
        h = 0 
        for i, v in enumerate(key):
            h += 31 * i + int(v)
        return h
    
    def _cus_hash_2(self, key):
        h = md5(key.encode()).digest().hex()
        return h
    
    def put(self, key, v):
        _k = self._cus_hash(key)
        self.ht_table[_k] = v
    

    def get(self, key):
        _k = self._cus_hash(key)
        try:
            return self.ht_table[_k]
        except KeyError:
            return None

    def remove(self, key):
        _k = self._cus_hash(key)
        del self.ht_table[_k]

    def __repr__(self, *args, **kwargs):
        return "自定义哈希表实现"
    