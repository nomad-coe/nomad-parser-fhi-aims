import bisect
import h5py
import numpy
import sys
import json
import re
import logging
from nomadcore import CompactSha

"""utilities to handle normalized run data stored in an hdf5 file

r=TrajUtils.openTrajFile("path/to/traj.info.hdf5")
cEnergy0=TrajUtils.maybeGet(r,['configs','main','cEnergyDft','#0','data'])
iLast=TrajUtils.maybeGet(r,['configs','main','cEnergyDft','nLocalIndex'])
if not i is None:
  last=TrajUtils.maybeGet(r,['configs','main','cEnergyDft',"#"+str(int(i)-1),'data'])
last2=TrajUtils.maybeGet(r,['configs','main','cEnergyDft',"#-1",'data'])
runWasOk = TrajUtils.maybeGet(r,['runDesc','cleanEnd']
"""

class PartiallyIndexedPropAccessor(object):
  """helper class to access and iterate on partially indexed properties stored in an hdf5 file"""
  def __init__(self,configPropAccessor,indexes):
    super(PartiallyIndexedPropAccessor,self).__init__()
    self.configPropAccessor=configPropAccessor
    self.indexes=indexes
    if self.configPropAccessor is None:
      return
    for iIndex, indexValue in enumerate(indexes):
      if self.configPropAccessor.indexes[iIndex]!=indexValue:
        raise Exception("inconsistent indexing in PartiallyIndexedPropAccessor, {0} {1}".format(
          self.configPropAccessor.indexes, indexes))

  def __iter__(self):
    return PartiallyIndexedPropAccessor(self.configPropAccessor, self.indexes)

  def valid(self):
    return not self.configPropAccessor is None

  def next(self):
    if self.configPropAccessor is None:
      raise StopIteration()
    res=self.configPropAccessor.next()
    if not self.indexes:
      return res
    if (not self.configPropAccessor.indexes or
        len(self.configPropAccessor.indexes)<len(self.indexes)):
      self.configPropAccessor=None
    for iIndex, indexValue in enumerate(self.indexes):
      if self.configPropAccessor.indexes[iIndex]!=indexValue:
        self.configPropAccessor=nextConfigIndex
    return res

  def __getitem__(self,index):
    configPropAccessor=self.configPropAccessor
    if configPropAccessor is None:
      return None
    if type(index) is slice:
      raise TypeError('slices not yet supported')
    else:
      i = index
    if type(i) is str or type(i) is unicode:
      try:
        i=int(i)
      except:
        pass
    if type(i) is int:
      if (configPropAccessor.nIndexes==0 or
          (self.indexes and len(self.indexes)==configPropAccessor.nIndexes)):
        # index of the local repetition
        lb,ub=configPropAccessor.indexRanges[-1]
        localI=configPropAccessor.localI+i
        if (localI<lb or ub<=localI):
          return None
        res=configPropAccessor.propValue(localI)
        return res
      if len(self.indexes)>configPropAccessor.nIndexes:
        raise Exception("too many indexes in PartiallyIndexedPropAccessor")
      iIndexNow=len(self.indexes)
      lb,ub=configPropAccessor.indexRanges[iIndexNow]
      localI=bisect.bisect_left(configPropAccessor.gIndexes[:,iIndexNow],i,lb,ub)
      if not (localI>=lb and localI<ub):
        return None
      newIndexes=configPropAccessor.gIndexes[localI,0:configPropAccessor.nIndexes]
      if newIndexes[iIndexNow]!=i:
        return None
      for iIndex, indexValue in enumerate(self.indexes):
        if newIndexes[iIndex]!=indexValue:
          return None
      newCPA=configPropAccessor.dup()
      newCPA.updateLocalI(localI)
      res=PartiallyIndexedPropAccessor(newCPA, self.indexes+[i])
      return res
    else:
      return None

  def __str__(self):
    if sys.version_info.major>2:
      return self.__unicode__()
    else:
      return self.__unicode__().encode("utf_8")

  def __unicode__(self):
    if self.configPropAccessor is None:
      return "None"
    return str(res)

class ConfigPropAccessor(object):
  """helper class to access and iterate on configuration properties stored in an hdf5 file"""
  def __init__(self,propGroup,localI=0):
    #use slots?
    super(ConfigPropAccessor,self).__init__()
    self.propGroup=propGroup
    gIndexes=propGroup["groupIndexes"]
    self.gIndexes=gIndexes
    self.nIndexes=gIndexes.shape[1]-2
    self.localI=localI
    self.indexRanges=[(0,gIndexes.shape[0])]
    if localI>=0 and localI <gIndexes.shape[0]:
      indexes=tuple(gIndexes[localI,0:self.nIndexes])
      for iIndex in range(self.nIndexes):
        oldBounds=self.indexRanges[iIndex]
        iAtt=indexes[iIndex]
        lb=bisect.bisect_left(self.gIndexes[:,iIndex],iAtt,oldBounds[0],oldBounds[1])
        ub=bisect.bisect_right(self.gIndexes[:,iIndex],iAtt,oldBounds[0],oldBounds[1])
        self.indexRanges.append((lb,ub))
    else:
      indexes=None
    self.indexes=indexes

  def dup(self):
    res=ConfigPropAccessor(self.propGroup,-1)
    res.localI=self.localI
    res.indexes=self.indexes
    res.indexRanges=list(self.indexRanges)
    return res

  def updateLocalI(self, newLocalI):
    if self.localI == newLocalI:
      return
    oldLocalI=self.localI
    self.localI=newLocalI
    newIndexRanges=[(0,gIndexes.shape[0])]
    if localI>=0 and localI<gIndexes.shape[0]:
      indexes=tuple(gIndexes[localI])
      self.indexes=indexes
      oldIndexes=self.indexes
      iRebuildFrom=0
      if oldIndexes:
        while (iRebuildFrom < len(oldIndexes) and iRebuildFrom<indexes
          and oldIndexes[iRebuildFrom]==indexes[iRebuildFrom]):
          newIndexRanges.append(self.indexRanges[iRebuildFrom])
          iRebuildFrom+=1
      for iIndex in range(iRebuildFrom,self.nIndexes):
        oldBounds=self.indexRanges[iIndex]
        iAtt=indexes[iIndex]
        lb=bisect.bisect_left(self.gIndexes[:,iIndex],iAtt,oldBounds[0],oldBounds[1])
        ub=bisect.bisect_right(self.gIndexes[:,iIndex],iAtt,oldBounds[0],oldBounds[1])
        self.indexRanges.append((lb,ub))
    else:
      self.indexes=None
      self.indexRanges=newIndexRanges

  def __iter__(self):
    return ConfigPropAccessor(self.propGroup, self.localI)

  def propValue(self, localI):
      """access to value at localI"""
      nLocalI=self.gIndexes.shape[0]
      if localI<0 or localI>=nLocalI:
        return None
      lb=localI
      nIndexes=self.nIndexes
      indexes=tuple(self.gIndexes[localI,0:nIndexes])
      dSet=self.propGroup["d"+str(self.gIndexes[localI,nIndexes])]
      if indexes==self.indexes:
        lb,ub=self.indexRanges[-1]
      else:
        while lb>0 and tuple(self.gIndexes[lb-1,0:nIndexes])==indexes:
          localI0-=1
        nextIndex=None
        ub=localI+1
        while ub<nLocalI and tuple(self.gIndexes[ub,0:nIndexes])==indexes:
          ub+=1
      nextIndex=None
      if ub<nLocalI:
        nextIndex=self.gIndexes[ub,0:self.nIndexes]
      prevIndex=None
      if lb>0:
        prevIndex=self.gIndexes[lb-1,0:self.nIndexes]
      el={
        'data':dSet[self.gIndexes[localI,nIndexes+1]],
        'indexes':indexes,
        'subIndex':localI-lb,
        'localIndex':localI,
        'nLocalIndex':nLocalI,
        'nSubIndex':ub-lb,
        'nextIndex':nextIndex,
        'previousIndex':prevIndex
      }
      return el

  def propValues(self, localI):
    """returns all the values starting at localI that share the same indexes"""
    res=[]
    nLocalI=self.gIndexes.shape[0]
    if localI<0 or localI>=nLocalI:
      return None
    firstVal=self.propValue(localI)
    for i in range(firstVal['nSubIndexes']-firstVal['subIndex']-1):
      res.append(self.propValue(localI+i))
    return res

  def next(self):
    res=self.propValues(self.localI)
    if not res:
      raise StopIteration()
    self.updateLocalI(self.localI+len(res))
    return res

  def __getitem__(self,index):
    nLocalI=self.gIndexes.shape[0]
    if isinstance(index,str) or isinstance(index,unicode) and index.startswith('#'):
      try:
        localI=int(index[1:])
        if localI<0:
          localI+=nLocalI
      except:
        pass
      else:
        return self.propValue(localI)
    if index == 'nLocalIndex':
      return nLocalI
    elif index == 'firstConfigIndex':
      if nLocalI == 0:
        return None
      return tuple(self.gIndexes[0,0:self.nIndexes])
    elif index == 'lastConfigIndex':
      if nLocalI == 0:
        return None
      return tuple(self.gIndexes[nLocalI-1,0:self.nIndexes])
    else:
      return PartiallyIndexedPropAccessor(self,[]).__getitem__(index)

  def keys(self):
    return ['nLocalIndex', 'firstConfigIndex', 'lastConfigIndex']

  def __str__(self):
    if sys.version_info.major>2:
      return self.__unicode__()
    else:
      return self.__unicode__().encode("utf_8")

  def __unicode__(self):
    res={}
    for k in self.keys():
      res[k]=self[k]
    return str(res)

def reprNoU(v):
  """representzation without u'' strings (plain quotes)"""
  if isinstance(v,unicode):
    return repr(v.encode("utf_8"))
  return repr(v)

global myRepr
myRepr=repr
if sys.version_info.major==2:
  myRepr=reprNoU

def findNextConfig(configsGroup, propertyNames, i):
  "finds the configuration after i across the given properties"
  res = None
  for pName in propertyNames:
    gIndexes=configsGroup[pName]["groupIndexes"]
    lb=bisect.bisect_left(gIndexes[:,0],i+1)
    if lb<gIndexes.shape[0]:
      if res == None:
        res=gIndexes[lb,0]
      elif gIndexes[lb,0] < res:
        res=gIndexes[lb,0]
  return res

def collectConfigProps(configsGroup, propertyNames, i):
  """returns a dictionary with all the properties of this configuration.
  Datasets are returned as h5py datasets (i.e. not expanded)"""
  if i is None:
    return None
  res = {}
  for pName in propertyNames:
    propGroup=configsGroup[pName]
    gIndexes=propGroup["groupIndexes"]
    lb=bisect.bisect_left(gIndexes[:,0],i)
    iVal=lb
    while True:
      if iVal>=gIndexes.shape[0] or gIndexes[iVal,0]!=i: break
      try:
        if pName in res:
          res[pName].append(propGroup['d'+str(gIndexes[iVal,1])][gIndexes[iVal,2]])
        else:
          res[pName]=[propGroup['d'+str(gIndexes[iVal,1])][gIndexes[iVal,2]]]
      except Exception, e:
        raise Exception("property:{}, gIndexes:{}, exception:{}".format(pName, gIndexes[:], e))
      iVal+=1
  return res

def lastDefinition(propGroup, i=None):
  """returns the last definition of the given property at i or before"""
  gIndexes=propGroup['groupIndexes']
  if i is None:
    if gIndexes.shape[0]>0:
      return gIndexes[gIndexes.shape[0]-1,0]
    return None
  lb=bisect.bisect_left(gIndexes[:,0],i)
  if lb<gIndexes.shape[0] and lb>=0:
    return gIndexes[lb,0]
  return None

def strToInt(x):
  if isinstance(x,str) or isinstance(x,unicode):
    try:
      return int(x)
    except:
      pass
  return x

def maybeGet(obj, path):
  """follows the path (sequence of indexes) given in path, returns None if it fails
  Nice to use, but relatively slow"""
  iPath=0
  lenPath=len(path)
  nrRe=re.compile(r"-?[0-9]+$")
  while True:
    try:
      while(iPath<lenPath):
        pathSegment=path[iPath]
        try:
          if type(obj) is h5py.Dataset:
            try:
              return obj[map(strToInt,path[iPath:])]
            except:
              pass
          obj = obj[pathSegment]
        except:
          #logging.getLogger("nomadcore.maybeGet").exception("in pathSegment %s",pathSegment)
          if ((isinstance(pathSegment,str) or isinstance(pathSegment,unicode))
              and nrRe.match(pathSegment)):
            obj = obj[int(pathSegment)]
          else:
            try:
              obj=obj.toDict()
            except:
              if isinstance(pathSegment,str) or isinstance(pathSegment,unicode):
                if pathSegment[:1]=='_':
                  raise Excption("Invalid private access")
                else:
                  obj = getattr(obj, pathSegment)
              else:
                raise
            else:
              obj=obj[pathSegment]
        iPath+=1
      if type(obj) is h5py.Group and 'lastDataset' in obj.attrs:
        return ConfigPropAccessor(obj)
      try:
        return obj.accessValue()
      except:
        return obj
    except:
      #logging.getLogger("nomadcore.maybeGet").exception("external handler ")
      if iPath < lenPath and path[iPath]=="config":
        if iPath+1 < lenPath:
          try:
            obj=config(obj,int(path[iPath+1]))
          except:
            return None
          iPath+=2
        else:
          return None
      elif type(obj) is h5py.Group and 'lastDataset' in obj.attrs:
        obj = ConfigPropAccessor(obj)
      else:
        return None

def openTrajFile(filePath):
  """opens an hdf5 trajectory file"""
  f=h5py.File(filePath,"r")
  res={'file':f}
  if 'type' in f.attrs:
    res['type']=f.attrs['type']
  if 'format' in f.attrs:
    res['format']=f.attrs['format']
    if re.match(r'Hdf5NomadInfoV1_[0-9]+', f.attrs['format']):
      if ('configs' in f):
        res['configs']=f['configs']
      if 'runDesc' in f.keys():
        res['runDesc']=json.loads(f['runDesc'][0])
      if 'keys' in f.keys():
        res['keys']=json.loads(f['keys'][0])
      if 'meta_keys' in f.keys():
        res['meta_keys']=json.loads(f['meta_keys'][0])
      if 'connection_from_to' in f.keys():
        res['connection_from_to']=f['connections_from_to']
      if 'connections_kind' in f.keys():
        res['connections_kind']=f['connections_kind']
      if 'connection_distance' in f.keys():
        res['connection_distance'] = f['connection_distance']
  return res

def config(configsGroup, i):
  """reads the configuration at index i in the given configuration group.
  Todo: add cell/periodicity info"""
  if i is None:
    lastDefinition(configsGroup["particlePositions"], i)
  if i is None:
    return None
  iNames=lastDefinition(configsGroup["particleKindNames"], i)
  pNames=collectConfigProps(configsGroup,["particleKindNames"],iNames)
  pPos=collectConfigProps(configsGroup,["particlePositions"],i)
  res={"configIndex":i, "inConfigIndex":0}
  res.update(pNames)
  res.update(pPos)
  if (len(res["particleKindNames"]) != 1 and
    len(res["particleKindNames"])!=len(res["particlePositions"])):
    raise Exception("particleKindNames is defined an unexpected number of times: {!r}".format(res))
  if len(particleKindNames)>1:
    j=1
    for pk in particleKindNames[1:]:
      if pk!=particleKindNames[0]:
        raise Exception("particleKindNames has different definitions at [{},0] and [{},{}]: {!r}".format(i,i,j,res))
      j+=1
  if len(particlePositions)>1:
    j=1
    for pk in particlePositions[1:]:
      if pk!=particlePositions[0]:
        raise Exception("particlePositions has different definitions at [{},0] [{},{}] : {!r}".format(i,i,j,res))
      j+=1
  return propGroup["particleKindNames"]
  return res

def configId(configDict):
  """returns a unique identification string for the given configuration"""
  sha = CompactSha.sha224()
  keys = list(configDict.keys())
  keys.sort()
  for name in keys:
    if name in ['configIndex', 'inConfigIndex', 'localIndex', 'nLocalIndex', 'nInConfigIndex',
                'nextConfigIndex', 'previousConfigIndex']:
      continue
    val = configDict[name]
    sha.update(name)
    if type(val) == numpy.ndarray:
      if val.dtype.kind=='f':
        sha.update(numpy.array_str(val, precision=6, suppress_small=true))
      else:
        sha.update(numpy.array_str(val))
    else:
      sha.update(repr(str))
  return sha.hex()

