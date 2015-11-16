#!/usr/bin/env python
import numpy
import sys, os, os.path
import re
import codecs
import json
import traceback
import shutil
import magic
from nomadcore import CompactSha

class Logger(object):
  def __init__(self,addMessage):
    self.addMessage=addMessage

  def addNote(self, message):
    "adds a note"
    return self.addMessage(message,0)

  def addWarning(self, message):
    "adds a warning"
    return self.addMessage(message,1)

  def addError(self, message):
    "adds a error"
    return self.addMessage(message,2)

  @staticmethod
  def labelForLevel(level):
    if level <= 0:
      return 'note'
    elif level <=1:
      return 'warning'
    else:
      return 'error'
  @staticmethod
  def formatMessage(msg, outF):
    label = "{:8}: ".format(msg.get('kind','message').upper())
    nChars=len(label)
    lineLen=88
    outF.write("\n")
    outF.write(label)
    sp=re.compile(r"(\s|\[|\]|[-.:,;+=]|\n)")
    for chunk in sp.split(msg.get('message','** no message **')):
      if chunk=="\n":
        outF.write(chunk)
        outF.write("    ")
        nChars=4
      elif nChars+len(chunk)<lineLen or nChars<lineLen/3 or sp.match(chunk):
        outF.write(chunk)
      else:
        outF.write("\n    ")
        outF.write(chunk)
        nChars=len(chunk)+4
    outF.write("\n")

def findEncoding(path, defaultEncoding="utf_8"):
  """tries to find the encodings of the file at the given path"""
  with open(path,"rb") as f:
    start = f.read(1024)
  encoding=None
  # find out encoding and file type
  boms={
    'utf_8':codecs.BOM_UTF8,
    'utf_16_be':codecs.BOM_UTF16_BE,
    'utf_16_le':codecs.BOM_UTF16_LE,
    'utf_32_be':codecs.BOM_UTF32_BE,
    'utf_32_le':codecs.BOM_UTF32_LE
  }
  def tryDecoding(encodingAtt,chunk=start):
    try:
      tmpDec=codecs.getincrementaldecoder(encodingAtt)("strict")
      tmpDec.decode(chunk)
      with open(path,"rb") as f2:
        chunk2 = f2.read(1024)
        while True:
          chunk2 = f2.read(1024)
          if not chunk2:
            break
          tmpDec.decode(chunk2)
        tmpDec.decode(b'', final=True)
      return encodingAtt
    except:
      return None
  if (defaultEncoding != None):
    encoding=tryDecoding(defaultEncoding)
  with magic.Magic(flags=magic.MAGIC_MIME_ENCODING) as m:
    encoding=m.id_filename(path)
  if encoding == 'binary':
    return (None,"")
  encoding=tryDecoding(encoding)
  if encoding == None:
    for ecodingAtt, bom in boms.items():
      if (start[0:len(bom)] == bom):
        if (tryDecoding(encodingAtt, start[len(bom):])):
          encoding=encodingAtt
  if (encoding == None):
    encoding=tryDecoding("utf_8")
  if (encoding == None and
     start[0]==b'\x00' and start[1]!=b'\x00' and
     start[2]==b'\x00' and start[3]!=b'\x00'):
    encoding=tryDecoding("utf_16_be")
  if (encoding == None and
     start[1]==b'\x00' and start[0]!=b'\x00' and
     start[3]==b'\x00' and start[2]!=b'\x00'):
    encoding=tryDecoding("utf_16_le")
  if (encoding == None and
     start[0]==b'\x00' and start[1]==b'\x00' and
     start[2]==b'\x00' and start[3]!=b'\x00' and
     start[4]==b'\x00' and start[5]==b'\x00' and
     start[6]==b'\x00' and start[7]!=b'\x00'):
    encoding=tryDecoding("utf_32_be")
  if (encoding == None and
     start[0]!=b'\x00' and start[1]==b'\x00' and
     start[2]==b'\x00' and start[3]==b'\x00' and
     start[4]!=b'\x00' and start[5]==b'\x00' and
     start[6]==b'\x00' and start[7]==b'\x00'):
    encoding=tryDecoding("utf_32_le")
  text=""
  if not encoding is None:
    i0=0
    if (encoding in boms):
      bom=boms[encoding]
      if (start[0:len(bom)] == bom):
        i0=len(bom)
    tmpDec=codecs.getincrementaldecoder(encoding)("strict")
    text=tmpDec.decode(start[i0:])
  return (encoding, text)

mimeType2Format={
  'application/x-xz':'XzCompressed',
  'application/x-gzip':'GzipCompressed',
  'application/x-bzip2':'Bzip2Compressed',
  'application/zip':'ZipArchive',
  'application/x-hdf':'Hdf5Document',
  'text/x-nomadInfo+json':'JsonNomadInfoV1_0',
  'application/x-nomad-traj+hdf5':'Hdf5NomadInfoV1_0',
  "text/x-json":'JsonDocument',
  "application/x-json":'JsonDocument',
  "application/xml":'XmlDocument',
  "text/xml":'XmlDocument',
  "application/html":'HtmlDocument',
  "text/html":'HtmlDocument'
}

class BasicFileInfo(object):
  """Basic informations on a file (type, format, gid)"""
  def __init__(self, localPath=None, sha224=None, sha512=None, mimeType=None, format=None,
      encoding=None,
      fileType=None, data=None):
    self.sha224=sha224
    self.sha512=sha512
    self.mimeType=mimeType
    self._format=format
    self.encoding=encoding
    self.data=data
    self._fileType=fileType
    self.localPath=localPath

  def fileType():
    doc = "The fileType, the name of the KindInfo of kind DocumentType of the file."
    def fget(self):
      if self._fileType is None:
        return self.format
      else:
        return self._fileType
    def fset(self, value):
      self._fileType = value
    return locals()
  fileType = property(**fileType())

  def format():
    doc = "The format of the file, described using the name of its KindInfo of kind DocumentType."
    def fget(self):
      if self._format is None:
        return 'UnknownDocumentType'
      else:
        return self._format
    def fset(self, value):
      self._format = value
    return locals()
  format = property(**format())

  @classmethod
  def fromMeta(cls, meta, localPath=None):
    """returns the basic meta informations of the file at the given path"""
    res=cls(localPath=localPath)
    if localPath and not ('fromat' in meta and 'fileType' in meta
        and 'mimeType' in meta and 'encoding' in meta):
      res.setTypeFromLocalFile()
    if 'encoding' in meta:
      res.format=meta['encoding']
    if 'format' in meta:
      res.format=meta['format']
    if 'fileType' in meta:
      res.fileType=meta['fileType']
    if 'mimeType' in meta:
      res.mimeType=meta['mimeType']
    if localPath and not ('sha224' in meta and 'sha512' in meta):
      res.setTypeFromLocalFile()
    if 'sha224' in meta:
      res.sha224=meta['sha224']
    if 'sha512' in meta:
      res.sha512=meta['sha512']
    return res

  @classmethod
  def fromLocalFile(cls, localPath):
    """returns the basic meta informations of the file at the given path"""
    res=cls(localPath=localPath)
    res.setFromLocalFile()
    return res

  def encodingKwds(self):
    res={}
    if sys.version_info.major>2 and self.encoding:
      res['encoding']=self.encoding
    return res

  def setTypeFromLocalFile(self, localPath=None):
    if localPath is None:
      localPath=self.localPath
    fileType=None
    mimeType=None
    if os.path.isdir(localPath):
      format='directory'
      fileType='directory'
    with magic.Magic(flags=magic.MAGIC_MIME_TYPE) as m:
      mimeType=m.id_filename(localPath)
    self.encoding, text=findEncoding(localPath)
    data=None
    format=None
    if mimeType in mimeType2Format:
      format=mimeType2Format[mimeType]
    elif mimeType.startswith('video'):
      format='VideoDocument'
    elif mimeType.startswith('image'):
      format='ImageDocument'
    elif mimeType == 'application/x-hdf':
      with h5py.File(localPath) as f:
        if 'format' in f.attrs and (isinstance(f.attrs['format'], str)
            or isinstance(f.attrs['format'], unicode)) and re.match('Hdf5NomadInfoV1_.*',f.attrs['format']):
          format = f.attrs['format']
          if 'type' in f.attrs:
            fileType=f.attrs['type']
        else:
          format = 'Hdf5Document'
    if format is None or format=='TextDocument' or format=='JsonDocument' and self.encoding:
      try:
        with open(localPath, "r", **self.encodingKwds()) as f:
          jsonData=json.load(f)
      except:
        pass
      else:
        data=jsonData
        format="JsonDocument"
        if (isinstance(jsonData,dict) and 'format' in jsonData
            and (isinstance(jsonData['format'], str) or isinstance(jsonData['format'], unicode))
            and re.match(r'JsonNomadInfoV1_[0-9]+', jsonData.get('format',''))):
          format=jsonData['format']
          fileType=jsonData.get('type', None)
          #try:
          #  fileTypeKindInfo=KindInfo.objects.get(name=fileType, kind=basicInfoKinds.shaof('DocumentType'))
          #except KindInfo.DoesNotExist:
          #  # try to add
          #  pass
          #if fileTypeKindInfo is None:
          #  metaKeys=InfoKindEnv(jsonData.get('meta_keys',[]))
          #  KindInfo.ensureInfoKindEnv(metaKeys)
          #fileTypeKindInfo=KindInfo.objects.get(
          #  name=fileType, kind=basicInfoKinds.shaof('DocumentType'))
          #  defaults={
          #    # gid = models.CharField(max_length=57, primary_key=True)
          #    # name = models.CharField(max_length=128,unique=True)
          #    # description = models.TextField(blank=True)
          #    # kind = models.ForeignKey("self", related_name="meta_childs", null=True)
          #    # units = models.CharField(max_length=128)
          #    # super_kind = models.ForeignKey("self", related_name="designated_super_of+", null=True)
          #    # super_kinds = models.ManyToManyField("self",symmetrical=False, related_name="sub_kinds")
          #    # cumulative_super_kinds = models.ManyToManyField("self",symmetrical=False, related_name="cumulative_sub+")
          #    # public = models.BooleanField(default=True)
          #    # publicOverride = models.ForeignKey("self", related_name="superseded_kinds")
          #  })
    self.mimeType=mimeType
    self.format=format
    self.fileType=fileType
    self.data=data

  def setShasFromLocalFile(self, localPath=None):
    if localPath is None:
      localPath=self.localPath
    if os.path.isdir(localPath):
      # recursive checksum not implemented
      return
    with open(localPath,"rb") as f:
      checkSum=CompactSha.sha224()
      checkSum2=CompactSha.sha512()
      while True:
        block=f.read(32*1024)
        if not block:
          break
        checkSum.update(block)
        checkSum2.update(block)
      self.sha224=checkSum.b64digest()+"F"
      self.sha512=checkSum2.b64digest()+"F"

  def setFromLocalFile(self,localPath=None):
    """Calculates the FileMetas of the given file"""
    if localPath is None:
      localPath=self.localPath
    self.setTypeFromLocalFile(localPath)
    self.setShasFromLocalFile(localPath)

def checkBaseFile(indexBuilder, path, metaInfo):
  "checks that the base file of a comment file exists"
  if (not os.path.exists(path[0:-len("comments.md")])):
    indexBuilder.addWarning("orfan comments file {0!r} ignored".format(path))

def jsonDumpStream(obj, fOut):
  """Dumps the object obj to an newly created utf_8 file at path"""
  json.dump(obj,fOut,sort_keys=True,indent=2,separators=(',', ': '),
      ensure_ascii=False,check_circular=False)

def jsonDump(obj, path):
  """Dumps the object obj to an newly created utf_8 file at path"""
  kwds=dict()
  if sys.version_info.major>2:
    kwds["encoding"]="utf_8"
  with open(path,"w",**kwds) as f:
    jsonDumpStream(obj,f)

class ShaStreamer(object):
  """a file like object that calculates one or more shas"""
  def __init__(self, shas=None):
    self.shas = shas
    if shaS is None:
      self.shas=(CompactSha.sha224(),)
  def write(self, val):
    for sha in self.shas:
      sha.update(val)
  def b64digests(self):
    return [sha.b64digest() for sha in self.shas]

def addShasOfJson(obj, shas=None):
  """adds the jsonDump of obj to the shas"""
  streamer=ShaStreamer(shas)
  jsonDumpStream(obj,streamer)
  return streamer

def normalizedJsonGid(obj, shas=None):
  """returns the gid of the standard formatted jsonDump of obj"""
  return map(lambda x: x+'j', addShasOfJson(shas).b64digests())

class DirToDo(object):
  "represent a directry that is being processed"
  def __init__(self, path, context, op):
    self.context=context
    self.path=path
    self.op=op
  def doOp(self,pWorker):
    self.op(self.path, self.context)

class PathWorker(object):
  """Works on fromPath, and puts the results into toPath

  currently this is iterative, but still does depth first traversal
  (keeps an explicit "stack" in pathsToHandle), this approach is more
  efficient.
  A simple recursion does have some advantages (exitDir is called
  after all subdirectories have been fully processed), but until now
  it was not needed."""

  def __init__(self, fromPath, toPath, acceptSamePaths=False,
    assimilationInfo={}, **kwrds):
    "builds an index of fromPath into toPath"
    super(PathWorker,self).__init__(**kwrds)
    self.fromPath = os.path.normpath(fromPath)
    self.toPath   = os.path.normpath(  toPath)
    p1=self.fromPath+'/'
    p2=self.toPath+'/'
    if ((p1.startswith(p2) or p2.startswith(p1)) and not acceptSamePaths):
      raise Exception("fromPath and toPath should not be nested")
    self.messages = []
    self.errorLevel = 0
    self.pathsToHandle = []
    self.logger=Logger(self.addMessage)
    self.assimilationInfo=assimilationInfo

  def addMessage(self, message, level=0):
    "adds a message"
    if self.errorLevel<level:
      self.errorLevel=level
    res={
      'kind':Logger.labelForLevel(level),
      'message':message
    }
    self.messages.append(res)
    return res

  def toTargetPath(self,path):
    """trasforms a path from fromPath into toPath"""
    return os.path.normpath(os.path.join(self.toPath,os.path.relpath(path,self.fromPath)))

  def enterDir(self,path):
    "called when entering a directory, should return a context for the directory"
    return {}

  def exitDir(self,path,context):
    "called when exiting from a directory"
    pass

  def handleDirPre(self,path,superContext):
    "handles a directory in a directory (before recursing into it)"
    pass

  def handleFile(self,path,context):
    "handles a file in a directory"
    pass

  def loopDir(self,path, superContext=None):
    """loops on the content of the given dir"""
    context=self.enterDir(path)
    self.pathsToHandle.append(DirToDo(path,context, self.exitDir))
    for f in os.listdir(path):
      pathAtt=os.path.join(path,f)
      if (os.path.isdir(pathAtt)):
        self.handleDirPre(pathAtt,context)
        self.pathsToHandle.append(DirToDo(pathAtt, context, self.loopDir))
      else:
        self.handleFile(pathAtt,context)

  def finish(self):
    pass

  def handleAll(self, superContext=None, startDir=None):
    "loops on all the files and directories in fromPath"
    if startDir is None:
      startDir=self.fromPath
    if os.path.isdir(startDir):
      self.pathsToHandle.append(DirToDo(startDir, superContext, self.loopDir))
    while len(self.pathsToHandle)>0:
      pathToDo=self.pathsToHandle.pop()
      pathToDo.doOp(self)
    self.finish()

  def dirSha(self, path, dirMeta=None):
    "sha224 and 512 of a directory that already has the sha224 and 512 for all its parts"
    if dirMeta is None:
      with open(self.toTargetPath(os.path.join(path,"meta.json")),"r") as f:
        dirMeta=json.load(f)
    idx=dirMeta['index']
    sortedFiles=list(idx.keys())
    sortedFiles.sort()
    checkSum=CompactSha.sha224()
    checkSum2=CompactSha.sha512()
    for fName in sortedFiles:
      if fName == 'meta.json':
        continue
      meta=idx[fName]
      if meta.get('fileType','')=='directory':
        with open(self.toTargetPath(os.path.join(os.path.join(path,fName),
            "meta.json")),"r") as f:
          meta=json.load(f).get('meta', {})
      if not 'sha224' in meta:
        raise Exception('index of {0} has no sha224 for {1}'.format(path, fName))
      if not 'sha512' in meta:
        raise Exception('index of {0} has no sha512 for {1}'.format(path, fName))
      delimitedName="{0}_{1}".format(len(fName),fName)
      checkSum.update(delimitedName)
      checkSum.update("57_")
      checkSum.update(meta['sha224'])
      checkSum2.update(delimitedName)
      checkSum2.update("129_")
      checkSum2.update(meta['sha512'])
    return {
      'sha224':checkSum.b64digest()+"D",
      'sha512':checkSum2.b64digest()+"D"
    }

  def shaOfPath(self,path):
    "returns the sha224 and 512 of the given path"
    meta = self.metaOfPath(path)
    return { 'sha224': meta['sha224'], 'sha512':meta['sha512'] }

  def sha224OfPath(self,path):
    "returns the sha224 of the given path"
    meta = self.metaOfPath(path)
    return meta['sha224']

class IndexBuilder(PathWorker):
  def __init__(self, fileTypes, overrideChecksums=True, overrideDates=True, mkSimlinks=True, **kwrds):
    "builds an index of fromPath into toPath"
    super(IndexBuilder,self).__init__(acceptSamePaths=True, **kwrds)
    self.overrideChecksums = overrideChecksums
    self.overrideDates = overrideDates
    self.mkSimlinks = mkSimlinks
    self.unknownFileTypes = []
    self.onChecksumChange = None
    self.fileTypes=fileTypes

  def enterDir(self, path):
    """tries to index the given directory"""
    dirMeta={}
    metaPath=os.path.join(path,"meta.json")
    if (os.path.isfile(metaPath)):
      try:
        kwds=dict()
        if sys.version_info.major>2:
          kwds["encoding"]=findEncoding(metaPath,"utf_8")[0]
        dirMeta=json.load(open(metaPath,"r",**kwds))
      except:
        self.logger.addWarning("error reading meta info from {0!r}: {1}".format(metaPath,sys.exc_info()))
    dirIndex={}
    if 'index' in dirMeta:
      dirIndex=dirMeta['index']
      if not isinstance(dirIndex,dict):
        addWarning("meta info 'index' at {0!r} does not parse as a dictionary but as {1}, ignoring".format(metaPath, type(dirIndex)))
        dirIndex={}
      else:
        for f in list(dirIndex.keys()):
          if not os.path.exists(os.path.join(path,f)):
            addWarning("removing not existing file info {0!r} from index at {1!r}".format(f, metaPath))
            del dirIndex[f]
    tPath=self.toTargetPath(path)
    if not os.path.isdir(tPath):
      os.makedirs(tPath)

    selfMeta=dirMeta.get('meta',{})
    if not isinstance(selfMeta,dict):
      addWarning("meta info for {0!r} is not a dict, but {1}, ignoring".format(path,type(selfMeta)))
      selfMeta={}
    stats = os.stat(path)
    if self.overrideDates or not 'mtime' in selfMeta:
      selfMeta['mtime']=stats.st_mtime
    if self.overrideDates or not 'ctime' in selfMeta:
      selfMeta['ctime']=stats.st_ctime
    commentsPath=os.path.join(path,"comments.md")
    if (os.path.isfile(commentsPath)):
      selfMeta['comments']=os.path.basename(commentsPath)
    selfMeta['format']="directory"
    selfMeta['fileType']="directory"
    dirMeta['meta']=selfMeta
    dirMeta['index']=dirIndex
    return dirMeta

  def handleDirPre(self, path, superContext):
      # real (complete) meta info is in the directory
     dirMeta=superContext['index'][os.path.basename(path)]={ 'fileType':"directory" }

  def exitDir(self,path,dirMeta):
    meta=dirMeta['meta']
    if self.overrideChecksums or not 'sha224' in meta or not 'sha512' in meta:
      shas=self.dirSha(path, dirMeta)
      if 'sha224' in meta and shas['sha224'] != meta['sha224']:
        self.onChecksumChange(path,meta,shas)
        if self.overrideChecksums:
          meta.update(shas)
        else:
          meta['sha512']=shas['sha512']
      elif 'sha512' in meta and shas['sha512'] != meta['sha512']:
        self.onChecksumChange(path,meta,shas)
        if self.overrideChecksums:
          meta.update(shas)
        else:
          meta['sha224']=shas['sha224']
      else:
       meta.update(shas)
    metaPath=os.path.join(path,"meta.json")
    jsonDump(dirMeta,self.toTargetPath(metaPath))

  def handleFile(self, path, dirMeta):
    """tries to find out the file type of the given file"""
    dirIndex=dirMeta.get('index',{})
    commentsPath=path+".comments.md"
    fileName = os.path.basename(path)
    metaInfo=dirIndex.get(fileName,{})
    metaInfo['fileName']=fileName
    stats = os.stat(path)
    if self.overrideDates or not 'mtime' in metaInfo:
      metaInfo['mtime']=stats.st_mtime
    if self.overrideDates or not 'ctime' in metaInfo:
      metaInfo['ctime']=stats.st_ctime
    if (os.path.isfile(commentsPath)):
      metaInfo['comments']=os.path.basename(commentsPath)
    if fileName in dirIndex:
      metaInfo.update(dirIndex[fileName])
    matchingTypes={}
    basicInfo=BasicFileInfo(localPath=path)
    basicInfo.setTypeFromLocalFile(path)
    mimeType=basicInfo.mimeType
    format=basicInfo.format
    encoding=basicInfo.encoding
    if not 'fileType' in metaInfo:
      encoding, text = findEncoding(path,metaInfo.get("encoding",encoding))
      for fType, fp in self.fileTypes.items():
        mimeTypeRe=fp.get('mimeTypeRe',None)
        if mimeTypeRe and not mimeTypeRe.match(basicInfo.mimeType):
          continue
        formats=fp.get('formats',None)
        if formats and not format in formats:
          continue
        if ('fileRe' in fp and fp['fileRe'].match(fileName)):
          pri=fp.get('filePri',0)
          mt=matchingTypes.get(pri,set())
          mt.add(fType)
          matchingTypes[pri]=mt
        if 'textRe' in fp:
          for m in fp['textRe'].findall(text):
            pri=fp.get('textPri',0)
            mt=matchingTypes.get(pri,set())
            mt.add(fType)
            matchingTypes[pri]=mt
            break
      k=list(matchingTypes.keys())
      k.sort(reverse=True)
      if len(k)>0:
        match=matchingTypes[k[0]]
        if len(match) != 1:
          addWarning("multiple matching types for file at {0!r}: priority:{1}, types:{!r}".format(path,k[0],list(match.keys())))
        for fType in match:
          metaInfo['fileType']=fType
          break
      else:
        metaInfo['fileType']=format
    metaInfo['format']=format
    fileInfo={}
    if 'fileType' in metaInfo:
      if metaInfo['fileType'] in self.fileTypes:
        fileInfo=self.fileTypes[metaInfo['fileType']]
        if 'metaInfoAdder' in fileInfo:
          fileInfo['metaInfoAdder'](self, path, metaInfo)
    else:
      self.unknownFileTypes.append(path)
    if self.overrideChecksums or not 'sha224' in metaInfo or not 'sha512' in metaInfo:
      with open(path,"rb") as f:
        checkSum=CompactSha.sha224()
        checkSum2=CompactSha.sha512()
        while True:
          block=f.read(32*1024)
          if not block:
            break
          checkSum.update(block)
          checkSum2.update(block)
        sha224=checkSum.b64digest()+"F"
        sha512=checkSum2.b64digest()+"F"
        if not self.onChecksumChange is None and (
            metaInfo.get("sha224",sha224)!=sha224 or
            metaInfo.get("sha512",sha512)!=sha512):
          onChecksumChange(path, metaInfo, {'sha224':sha224, 'sha512':sha512})
        if self.overrideChecksums or not 'sha224' in metaInfo:
          metaInfo["sha224"]=sha224
        if self.overrideChecksums or not 'sha512' in metaInfo:
          metaInfo["sha512"]=sha512
    dirIndex[fileName]=metaInfo
    if self.mkSimlinks and fileInfo.get('makeLink',True):
      target=self.toTargetPath(path)
      if path != target:
        os.symlink(os.path.abspath(path),target)
        kwds=dict()
        if sys.version_info.major>2:
          kwds["follow_symlinks"]=True
        if "mtime" in metaInfo:
          os.utime(target,(-1,metaInfo["mtime"]),**kwds)

  def finish(self):
    self.unknownFileTypes.sort()
    self.logger.addNote("Files without type: {0}".format(self.unknownFileTypes))

class Normalizer(PathWorker):
  """Normalizes the data in the indexed fromPath into toPath"""

  def __init__(self, fileNormalizers, **kwrds):
    super(Normalizer,self).__init__(**kwrds)
    self.metaCache={}
    self.fileNormalizers=fileNormalizers
    self.fromDirSha=None

  jsonDump=staticmethod(jsonDump)

  def dirMetaOfPath(self,path):
    if path in self.metaCache:
      return self.metaCache[path]
    if os.path.isdir(path):
      dirMeta={}
      try:
        kwds=dict()
        if sys.version_info.major>2:
          kwds["encoding"]="utf_8"
        with open(os.path.join(path,"meta.json"),"r",**kwds) as f:
          dirMeta=json.load(f)
      except:
        #self.logger.addWarning("failed to load meta info for {!r} due to exception:{}".format(
        #  path, traceback.format_exc()))
        pass
      self.metaCache[path]=dirMeta
      return dirMeta
    dirPath=os.path.dirname(path)
    if not os.path.isdir(dirPath):
      return {}
    dirMeta=self.dirMetaOfPath(dirPath)
    return dirMeta

  def metaOfPath(self,path):
    dirMeta=self.dirMetaOfPath(path)
    if os.path.isdir(path):
      return dirMeta.get('meta',{})
    idx=dirMeta.get('index',{})
    return idx.get(os.path.basename(path))

  def updateDirMetaOfPath(self,path,newMeta):
    dirPath=path
    if not os.path.isdir(path):
      dirPath=os.path.dirname(path)
    jsonDump(newMeta,os.path.join(dirPath,"meta.json"))

  def updateMetaOfPath(self,path,newMeta):
    dirMeta=self.dirMetaOfPath(path)
    if not 'index' in dirMeta:
      dirMeta['index']={}
    if not os.path.isdir(path):
      dirMeta['index'][os.path.basename(path)]=newMeta
    else:
      dirMeta['meta']=newMeta
    self.updateDirMetaOfPath(path,newMeta)

  def enterDir(self, path):
    return self.dirMetaOfPath(path).get('index',{})

  def handleFile(self,path,dirIndex):
    fName=os.path.basename(path)
    fileType=dirIndex.get(fName,{}).get('fileType','')
    if fileType in self.fileNormalizers.keys():
      self.fileNormalizers[fileType](self,path)

  def createDirectoryForPath(self,path, dirKind):
    """creates a directory for a dirKind (typically run or snapshot)

    The directory is target path of path with dirKind-Nr.
    Also sets up meta info both ways"""
    dirPath=path
    if not os.path.isdir(path):
      dirPath=os.path.dirname(path)
    targetDirPath=os.path.join(self.toTargetPath(dirPath),dirKind+'-')
    i=0
    while True:
      if not os.path.exists(targetDirPath+str(i).zfill(3)):
        targetDirPath+=str(i).zfill(3)
        break
      i+=1
    os.makedirs(targetDirPath)
    baseMeta=self.metaOfPath(path)
    if not dirKind in baseMeta:
      baseMeta[dirKind]=[]
    # use path relative to toPath instead???
    baseMeta[dirKind].append(os.path.relpath(targetDirPath,path))
    self.updateMetaOfPath(path,baseMeta)
    targetMeta=self.metaOfPath(targetDirPath)
    targetMeta['origin']=path
    self.updateMetaOfPath(targetDirPath,targetMeta)
    return targetDirPath

class Loader(Normalizer):
  """Loads data into the database"""

  def __init__(self, fromPath, toPath, fileLoaders, dirLoaders, **kwds):
    super(Loader,self).__init__(fromPath=fromPath, toPath="/tmp", acceptSamePaths=True,
      fileNormalizers=None, baseUrl=None, **kwds)
    self.fileLoaders=self.fileLoaders
    self.dirLoaders=dirLoaders
    self.baseUrl=baseUrl
    if len(baseUrl)>0 and baseUrl[-1] != '/':
      baseUrl+='/'

  def urlForPath(self,path):
    return baseUrl + os.path.relpath(path,self.fromPath)

  def toTargetPath(self,path):
    raise Exception("Loader has no target path")

  jsonDump=staticmethod(jsonDump)

  def enterDir(self, path):
    for handler in self.dirLoaders:
      handler(self,path=path,fileType=fileType)
    return self.dirMetaOfPath(path).get('index',{})

  def handleFile(self,path,dirIndex):
    fName=os.path.basename(path)
    fMeta=dirIndex.get(fName,{})
    basicInfo=BasicFileInfo.fromMeta(fMeta, path)
    m=None
    handled=0
    for handler in self.fileLoaders:
      if handled == 0 or handler.get('multiHandler', False):
        formatRe=handler.get('formatRe',None)
        if formatRe and not formatRe.match(basicInfo.format):
          continue
        fileTypeRe=handler.get('fileTypeRe',None)
        if fileTypeRe and not fileTypeRe.match(basicInfo.fileType):
          continue
        fileRe=handler.get('fileRe',None)
        if fileRe and not fileRe.match(fName):
          continue
        if formatRe or fileTypeRe or fileRe or handler.get('anyFile', False):
          if handler['handler'](self,basicInfo, fromUrl=self.urlForPath(path)):
            handled+=1

class DBAssimilator(PathWorker):
  """Assimilator a directory to the database"""

  def __init__(self, accessGroup, fileLoaders, dirLoaders, logFile, baseUrl,
    fileNormalizers, overrideChecksums=True, overrideDates=False,
    **kwds):
    super(DBAssimilator,self).__init__(**kwds)
    self.accessGroup=accessGroup
    self.fileLoaders=fileLoaders
    self.dirLoaders=dirLoaders
    self.logFile=logFile
    self.overrideDates=overrideDates
    self.overrideChecksums=overrideChecksums

  def addMessage(self, msg, level=0):
    if self.echoMessages:
      Logger.formatMessage(
        super(Assimilator,self).addMessage(msg,level),
        self.logFile)

  def handleAll(self):
    errorLevel=0
    assimilator=Assimilator(fileTypes=self.fileTypes,fromPath=self.fromPath,toPath=self.toPath,
      fileNormalizers=self.fileNormalizers, scriptingFiles=scriptingFiles, echoMessages=True)
    try:
      assimilator.handleAll()
    except:
      self.logger.addError("Indexing failure:" + traceback.format_exc())
    indexer=IndexBuilder(fileTypes=self.fileTypes,fromPath=self.fromPath,toPath=self.fromPath,
        overrideChecksums=self.overrideChecksums, overrideDates=self.overrideDates)
    try:
      indexer.handleAll()
      jsonDumpStream(indexer.messages,self.logFile)
      if indexer.errorLevel>0:
        self.logger.addMessage('Indexing had {0}s.'
          .format(Logger.labelForLevel(indexer.errorLevel)),
          indexer.errorLevel)
      else:
        self.logger.addNote("Indexing was successful.")
    except:
      self.logger.addError("Indexing failure:" + traceback.format_exc())
    if errorLevel>1:
      return False
    loader=Loader(fileLoaders=self.fileLoaders, dirLoaders=self.dirLoaders,
       fromPath=assimilator.rawDataPath, baseUrl=self.baseUrl)
    try:
      loader.handleAll()
      jsonDumpStream(loader.messages,self.logFile)
      if loader.errorLevel>0:
        self.logger.addMessage('Loading in db had {0}s.'
          .format(Logger.labelForLevel(loader.errorLevel)),
          loader.errorLevel)
      else:
        self.logger.addNote("Loading in db was successful.")
    except:
      self.logger.addError("Indexing failure:" + traceback.format_exc())

class Assimilator(PathWorker):
  """Assimilator tries to assimilate a directory

  the resulting directory uses a symlink for the raw data"""

  def __init__(self, fileTypes, fileNormalizers, scriptingFiles, echoMessages=True, **kwds):
    super(Assimilator,self).__init__(**kwds)
    self.echoMessages=echoMessages
    self.fileTypes=fileTypes
    self.fileNormalizers=fileNormalizers
    self.scriptingFiles=scriptingFiles

  def addMessage(self, msg, level=0):
    if self.echoMessages:
      Logger.formatMessage(
        super(Assimilator,self).addMessage(msg,level),
        sys.stdout)

  def handleAll(self):
    errorLevel=0
    commentsDir=os.path.join(self.toPath,"comments")
    os.makedirs(commentsDir)

    scriptDir=os.path.join(self.toPath,"comments/script")
    os.makedirs(scriptDir)
    lenBaseScriptDir=os.path.commonprefix(self.scriptingFiles).rfind("/")+1
    for relativeTargetPath,sourceF in self.scriptingFiles.items():
      targetPath=os.path.join(scriptDir,relativeTargetPath)
      targetDir=os.path.dirname(targetPath)
      if not os.path.exists(targetDir):
        os.makedirs(targetDir)
      shutil.copy2(sourceF,targetPath)

    rawDataPath=os.path.join(self.toPath,"raw_data")
    self.rawDataPath=rawDataPath
    indexer=IndexBuilder(fileTypes=self.fileTypes,fromPath=self.fromPath,toPath=rawDataPath)

    indexLogPath=os.path.join(commentsDir,"indexingLog.json")
    self.indexLogPath=indexLogPath
    try:
      indexer.handleAll()
      if indexer.errorLevel>0:
        self.logger.addMessage('Indexing had {0}s, detailed log in {1!r}.'
          .format(Logger.labelForLevel(indexer.errorLevel),indexLogPath),
          indexer.errorLevel)
      else:
        self.logger.addNote("Indexing was successful, detailed log in {0!r}."
          .format(indexLogPath))
    except:
      self.logger.addError("Indexing failure:" + traceback.format_exc())
    jsonDump(indexer.messages,indexLogPath)
    if errorLevel>1:
      return

    normalizedDataPath=os.path.join(self.toPath,"normalized_data")
    self.normalizedDataPath=normalizedDataPath
    normalizer=Normalizer(fileNormalizers=self.fileNormalizers, fromPath=rawDataPath,toPath=normalizedDataPath)
    normalizerLogPath=os.path.join(commentsDir,"normalizingLog.json")
    self.normalizerLogPath=normalizerLogPath
    normalizerIndexer=IndexBuilder(fileTypes=self.fileTypes,fromPath=normalizedDataPath,toPath=normalizedDataPath)
    try:
      normalizer.handleAll()
      normalizerIndexer.handleAll()
      if normalizer.errorLevel>0:
        self.logger.addMessage('Normalization had {0}s, detailed log in {1!r}.'
          .format(Logger.labelForLevel(normalizer.errorLevel),normalizerLogPath),
          normalizer.errorLevel)
      else:
        self.logger.addNote("Normalization was successful, detailed log in {0!r}."
          .format(normalizerLogPath))
    except:
      self.logger.addError("Normalization failure:" + traceback.format_exc())
    jsonDump(normalizer.messages,normalizerLogPath)
