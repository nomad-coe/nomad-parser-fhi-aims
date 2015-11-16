
# Author: Fawzi Mohamed

import numpy
import os, sys
import re
import traceback
import hashlib
import logging

from ase.atoms import Atoms

nLog = logging.getLogger("nomad")
#nLog.setLevel(logging.DEBUG)
#h = logging.StreamHandler(stream=sys.stdout)
#h.setLevel(logging.DEBUG)
#nLog.addHandler(h)


def maybeGet(obj, path):
    """follows the path (sequence of indexes) given in path, returns None if it fails
    Nice to use, but relatively slow"""
    iPath = 0
    lenPath = len(path)
    nrRe = re.compile(r"-?[0-9]+$")
    while True:
        try:
            while(iPath < lenPath):
                pathSegment = path[iPath]
                try:
                    #if type(obj) is h5py.Dataset:
                    #    try:
                    #        return obj[map(strToInt, path[iPath:])]
                    #    except:
                    #        pass
                    obj = obj[pathSegment]
                except:
                    #logging.getLogger("nomad").exception("in pathSegment %s",pathSegment)
                    if ((isinstance(pathSegment, str) or isinstance(pathSegment, unicode)) and nrRe.match(pathSegment)):
                        obj = obj[int(pathSegment)]
                    else:
                        try:
                            obj = obj.toDict()
                        except:
                            if isinstance(pathSegment, str) or isinstance(pathSegment, unicode):
                                if pathSegment[:1] == '_':
                                    raise Exception("Invalid private access")
                                else:
                                    obj = getattr(obj, pathSegment)
                            else:
                                raise
                        else:
                            obj = obj[pathSegment]
                iPath += 1
            #if type(obj) is h5py.Group and 'lastDataset' in obj.attrs:
            #    return ConfigPropAccessor(obj)
            try:
                return obj.accessValue()
            except:
                return obj
        except:
            #logging.getLogger("nomad").exception("external handler ")
            if iPath < lenPath and path[iPath]=="config":
                if iPath+1 < lenPath:
                    try:
                        obj = config(obj, int(path[iPath+1]))
                    except:
                        return None
                    iPath += 2
                else:
                    return None
            #elif type(obj) is h5py.Group and 'lastDataset' in obj.attrs:
            #    obj = ConfigPropAccessor(obj)
            else:
                return None

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


class ParserControl(object):
    "Class for parser control values"
    Open=1
    Closed=2
    CloseNormal=3
    CloseForced=4
    MaybeMatch=5
    Failed=6
    AtEnd=7

class ParseMode(object):
    """just an enum like class for parse modes"""
    Normal=1
    Speculative=2
    Confirmed=3

class CloseKind(object):
    """just an enum of close kinds"""
    Normal=1
    Forced=2

class MatchKind(object):
    """just an enum of LineParser match kinds"""
    Weak = 1 # weak match (should not highjack parsing if not current)
    # tier 1 rematch
    Sequenced = 2 # part of the sequence when going to the next match
    AlwaysActive = 4 # the match should always be picked up (even if far from last match)
    # tier 3 rematch
    Fallback = 8 # match to be executed when Sequenced and AlwaysActive rematch fail

    Repeating = 16 # the section is expected to repeat

    Floating = 32 # the match should "float" to the top of the stck without unwinding it
    Repeatable = 64 # can match more than once in its supersection
    LastNotEnd = 128 # if matching the last Sequential parser does not imply end
    Required = 256 # if a successful match is required when closing the parent section
    DefaultSet = 512 # not explicitly set
    ExpectFail = 1024 # the matching is expected to fail

class PropertyKind(object):
    commonProperty="CommonProperty"
    configProperty="ConfigProperty"
    section="Section"

class Property(object):
    registry={}
    usedProps=set()

    @classmethod
    def register(cls, prop):
        if prop.name in cls.registry:
            raise Exception("double registration of property " + repr(prop.name))
        cls.registry[prop.name]=prop

    @classmethod
    def withName(cls, name):
        return cls.registry.get(name, None)

    def __init__(self, name, description, superKeys, kind, units=None, dtype=None, repeats=False, contextName=None, shape=()):
        super(Property, self).__init__()
        self.name=name
        self.kind=kind
        self.description=description
        if superKeys is None:
            self.superKeys=[]
        else:
            self.superKeys=superKeys
        self._gid=None
        self.units=units
        self.dtype=dtype
        self.repeats=repeats
        self.shape=shape
        self.contextName=contextName
        self.register(self)
        for k in self.superKeys:
            if not k in self.registry:
                nLog.warn("property %r has non registered superKey %r",self.name, k)

    def gid(self):
        if not self._gid is None:
            return self._gid
        s224=hashlib.sha224()
        descVal=self.toDict(False)
        keys=list(descVal.keys())
        keys.remove('superKeys')
        for k in keys:
            v=str(descVal[k])
            s224.update(str(len(v)))
            s224.update(" ")
            s224.update(v)
        for superKName in self.superKeys:
            p=Property.withName(superKName)
            superG=p.gid()
            s224.update(superG)
        return s224.hexdigest()+'P'


    def prop(self):
        return self

    @staticmethod
    def strToValKind(dtype, strVal, logger, contextName=""):
        if strVal is None:
            strVal=""
        intRe=re.compile(r"[0-9]+$")
        doubleRe=re.compile(r"[-+]?[0-9]*\.[0-9]+(?:(?P<expChar>[eEdD])[+-]?[0-9]+)?$")
        if dtype is float:
            return float(re.compile("[dD]").sub("e",strVal))
        if dtype is None:
            if intRe.match(strVal):
             return int(strVal)
            m=doubleRe.match(strVal)
            if m and m.group("expChar"):
                strVal=strVal.replace(m.group("expChar"),"e")
                return float(strVal)
            return strVal
        if dtype is bool:
            return m.match(r"(?i)\s*(?:\.?f(?:alse)?\.?|n(?:o)|0)\s*", strVal) == None # to do (check)
        if dtype is numpy.ndarray:
            return numpy.array(
                map(lambda x:float(re.compile("[dD]").sub("e",x)),
                        strVal.split()))
        try:
            return dtype(strVal)
        except:
            logger.addWarning("{0} could not convert {1} to {2}".format(contextName,strVal,dtype))
        return strVal

    def strToVal(self, strVal, logger):
        return self.strToValKind(self.dtype, strVal, logger=logger, contextName=self.name)

    def transformValue(self, value, logger):
        return value

    def allSuperSections(self,res=None):
        if res is None:
            res=[]
        myK=[]
        for s in self.superKeys:
            if not s in res:
                res.append(s)
                myK.append(s)
        for s in myK:
            sKey=self.withName(s)
            if sKey is None:
                raise Exception("failed to find super key {0}".format(s))
            sKey.allSuperSections(res)
        return res

    def singleSuperSections(self,res=None):
        if res is None:
            res=[]
        if self.superKeys:
            kAtt=self.superKeys[0]
            if not kAtt in res:
                res.append(kAtt)
                sKey=self.withName(kAtt)
                if sKey is None:
                    raise Exception("failed to find super key {0}".format(kAtt))
                sKey.singleSuperSections(res)
        return res

    def singleSuperSectionVal(self, rootSectionVal, logger):
        section=rootSectionVal
        superSections=self.singleSuperSections()
        for i in range(len(superSections)-1,-1,-1):
            superSection=superSections[i]
            sectionInfo=Property.withName(superSection)
            if not superSection in section:
                if sectionInfo.repeats:
                    logger.addWarning("Repeating super section {0} was not present when adding {1}".format(superSection,self.name))
                    section[superSection]=[{}]
                else:
                    section[superSection]={}
            section=section[superSection]
            if sectionInfo.repeats:
                section=section[-1]
        return section

    def detailedName(self):
        return self.name

    def addConfigProperty(self, logger, globalContext, value):
        self.usedProps.add(self)
        i, dSet = globalContext.context.datasetForProperty(self,numpy.shape(value))
        nLog.debug("%s %s sets value %s %s %r %r",self.name,self.dtype,value,type(value), dSet, i)
        dSet[i]=value

    def handleProperty(self, logger, globalContext, localContext, value=None):
        try:
            nLog.debug("handleProperty %s, kind: %s", self.detailedName(),self.kind)
            context=globalContext.context.runInfo
            if not self.contextName is None:
                context=localContext
                while not context is None and context.section.name != self.contextName:
                    context=context.superSection
                if context is None:
                    logger.addWarning("Property {0} could not find context {1}, ignoring".format(self.name, self.contextName))
                    return
                context=context.context
            else:
                self.__class__.usedProps.add(self)
                globalContext.context.runInfoProps.add(self.name)
            #if self.kind==PropertyKind.configProperty:
            #    self.addConfigProperty(logger,globalContext,value)
            if self.kind==PropertyKind.commonProperty:
                section=self.singleSuperSectionVal(context, globalContext.context.logger)
                if self.repeats:
                    if not self.name in section:
                        section[self.name]=[]
                    section[self.name].append(value)
                else:
                    if self.name in section:
                        logger.addWarning("Non repeated Property {0} has been repeated, dropping previous value {1}".format(self.detailedName(), section[self.name]))
                    section[self.name]=value
            elif self.kind==PropertyKind.configProperty:
                self.addConfigProperty(logger,globalContext,value)
            elif self.kind==PropertyKind.section:
                if value is None:
                    value={}
                section=self.singleSuperSectionVal(context,globalContext.context.logger)
                if self.repeats:
                    if not self.name in section:
                        section[self.name]=[value]
                    else:
                        section[self.name].append(value)
        except:
            raise Exception("Exception while handling property {0} at {1}: {2}".format(self.name, globalContext.lineFile.posStr(), traceback.format_exc()))

    def toInfoKindEl(self):
        InfoKindEl(name=self.name, kindStr=self.kindStr, description=self.description,
            superKeys=self.superKeys, units=self.units)

    def toDict(self, addGid=True):
        res={
            'name':self.name,
            'kindStr':self.kind,
            'description':self.description,
            'superKeys':self.superKeys,
            'units':self.units,
            'dtype':str(self.dtype),
            'shape':self.shape,
            'repeats':self.repeats,
        }
        if addGid:
            res['gid']=str(self.gid())
        return res

    @staticmethod
    def sortAndCompleteProps(propsToSort):
        """builds a list of properties in propsToSort, so that all the dependecies of each
        property are present before them"""
        toDo=list(propsToSort)
        done=set()
        deps=[]
        res=[]
        while len(toDo)>0:
            pAtt=toDo.pop()
            nameAtt=pAtt.name
            if nameAtt in done:
                continue
            deps=[nameAtt]
            while len(deps)>0:
                nameAtt=deps[-1]
                pAtt=Property.withName(nameAtt)
                if pAtt is None:
                    raise Exception("missing dependent Property {0} following chain {1}".format(nameAtt, pAtt))
                hasDepsToDo=False
                for superName in pAtt.superKeys:
                    if not superName in done:
                        if superName in deps:
                            raise Exception("circular dependency {0}, {1}".format(deps,superName))
                        deps.append(superName)
                        hasDepsToDo=True
                        break
                if not hasDepsToDo:
                    deps.pop()
                    res.append(pAtt)
                    done.add(nameAtt)
        return res

    @classmethod
    def printProps(self,outPath,propsToPrint=None):
        if propsToPrint is None:
            propsToPrint=self.usedProps
        p=self.sortAndCompleteProps(propsToPrint)
        toP=map(lambda x:x.toDict(),p)
        jsonDump(toP, outPath)

    @classmethod
    def printInfoKinds(self,outPath,propsToPrint=None):
        if propsToPrint is None:
            propsToPrint=self.usedProps
        p=self.sortAndCompleteProps(propsToPrint)
        toP=map(lambda x:x.toInfoKindEl(),p)
        infoKinds=InfoKindEnv(toP)
        with open(outPath,"w") as f:
            infoKinds.serialize(f.write)

class Prop(Property):
    def __init__(self, name, description, superKeys=None, kind=None, **kwds):
        if kind is None:
            if name[0].isupper():
                kind=PropertyKind.section
            else:
                kind=PropertyKind.commonProperty
        super(Prop, self).__init__(name=name, description=description, superKeys=superKeys,
            kind=kind, **kwds)

class PropertyTransformer(object):
    """A simple value tranformer (for example to perform unit
        conversions)"""
    def __init__(self,basePropName,transform):
        self.basePropName=basePropName
        self.transform=transform

    def prop(self):
        p=Property.withName(self.basePropName)
        if p is None:
            raise Exception("PropertyTransformer refers to non existing property {0}".format(self.basePropName))
        return p

    def strToVal(self, strVal, logger):
        res=self.prop().strToVal(strVal, logger)
        return self.transformValue(res, logger)

    def transformValue(self, value, logger):
        return self.transform(value)

    def addConfigProperty(self, logger, globalContext, value):
        self.prop().addConfigProperty(logger, globalContext, value)

    def handleProperty(self, logger, globalContext, localContext, value=None):
        self.prop().handleProperty(logger, globalContext, localContext,value)

# LineParser interface:
# self.name
# self.kind
# self.subParsers
# def parse(self, parser, sectionContext, mode):
# def close(self,parser,sectionContext,closeKind):
# def rematch(self,parser,sectionContext, sectionIndex=-1, matchKind=MatchKind.Normal):

class LineParser(object):
    """Parses a line (or more) of a file"""
    def __init__(self, name, startReStr=None, subParsers=None,
        kind=MatchKind.Sequenced | MatchKind.DefaultSet,
        propNameMapper=None, localProps=None):

        super(LineParser,self).__init__()
        self.name=name

        if startReStr:
            self.startRe=re.compile(startReStr)
        else:
            self.startRe=None
        self.subParsers=subParsers
        if subParsers is None:
            self.subParsers=[]
        self.localProps=localProps
        if localProps is None:
            self.localProps={}
        self.nRecoverTop=-1
        self.nRecoverNormal=2
        self.kind=kind
        self.propNameMapper=propNameMapper

    def prop(self,sectionContext,name=None):
        if name is None:
            name=self.name
        p=None
        if not self.propNameMapper is None:
            name2=self.propNameMapper(name)
            if name2 in self.localProps:
                p=self.localProps[name2]
            elif p is None:
                if not sectionContext.superSection is None:
                    p=sectionContext.superSection.section.prop(sectionContext.superSection,name2)
                else:
                    p=Property.withName(name2)
        if p is None:
            if name in self.localProps:
                p=self.localProps[name]
            elif not sectionContext.superSection is None:
                p=sectionContext.superSection.section.prop(sectionContext.superSection,name)
            else:
                p=Property.withName(name)
        return p

    def handleMatch(self, matchDict, parser, sectionContext):
        for k, v in matchDict.items():
            p = self.prop(sectionContext, k)
            if p is None:
                parser.logger.addWarning("LineParser {0} has parsed group {1} without corresponding property, ignoring {2}".format(self.name, k, v))
            else:
                p.handleProperty(logger=parser.logger, globalContext=parser, localContext=sectionContext, value=p.strToVal(v, parser.logger))
        return True

    def open(self,parser,sectionContext,mode):
        pass

    def parse(self, parser, sectionContext, mode):
        if not self.startRe:
            return (ParserControl.Failed,sectionContext)
        m = self.startRe.match(parser.lineFile.line)
        if not m:
            return (ParserControl.Failed,sectionContext)
        if mode == ParseMode.Speculative and (self.kind & MatchKind.Weak) != 0:
            return (ParserControl.MaybeMatch,sectionContext) # do this check at the beginning?
        matchProps = m.groupdict()
        if len(self.subParsers)>0:
            if (mode in [ParseMode.Speculative, ParseMode.Confirmed] and
                    not sectionContext.tmpContext):
                sectionContext.close(parser,CloseKind.Forced)
            self.open(parser,sectionContext=sectionContext,mode=mode)
        if not self.handleMatch(matchProps, parser, sectionContext=sectionContext):
            return (ParserControl.Failed,sectionContext)
        if len(self.subParsers)>0:
            return (ParserControl.Open,sectionContext)
        return (ParserControl.Closed,sectionContext)

    def matchEnd(self,parser, sectionContext):
        return False

    def rematch(self,parser,sectionContext, lastSection,matchKind):
        """tries to find what matches the current line.
        Used when a mismatch is present, and when skipping lines."""
        nLog.debug("rematch %s %r %r %r",self.name, stackNames(sectionContext),stackNames(lastSection),matchKind)
        subIndex=sectionContext.subParserIndex
        iChecked=[0,0]
        if not (subIndex is None and subIndex >= -1 and (subIndex < len(self.subParsers) or (self.kind&MatchKind.LastNotEnd)!=0)):
            nChecks=self.nRecoverNormal
            nSubParsers=len(self.subParsers)
            if lastSection and lastSection.superSection is sectionContext:
                nChecks=self.nRecoverTop
            if nChecks == -1:
                nChecks+=nSubParsers
            startIdx=1
            if subIndex== -1:
                startIdx=0
            for i in range(startIdx,len(self.subParsers)):
                if (i%2==1):
                    idxAtt=(subIndex+(i+1)//2)%nSubParsers
                else:
                    idxAtt=(subIndex-i//2)%nSubParsers
                parserAtt=self.subParsers[idxAtt]
                pKind=parserAtt.kind
                if (pKind&matchKind == 0 or (pKind&MatchKind.Weak)!=0 or
                        ((pKind&matchKind&(MatchKind.AlwaysActive|MatchKind.Fallback))==0 and iChecked[i%2] >= nChecks and (pKind&MatchKind.Sequenced)!=0)
                        or ((pKind&(MatchKind.Repeatable|MatchKind.Repeating))==0 and parserAtt in sectionContext.matched)):
                    continue
                iChecked[i%2]+=1
                ctxAtt=SectionContext(parserAtt, superSection=sectionContext, tmpContext=True)
                (parseRes,newCtx)=ctxAtt.parse(parser=parser,mode=ParseMode.Speculative)
                nLog.debug("parsing of %s did yield %s",parserAtt.name,parseRes)
                if (parseRes in [ParserControl.Open, ParserControl.Closed, ParserControl.CloseNormal, ParserControl.CloseForced]):
                    ctxAtt.tmpContext=False
                    if ((parserAtt.kind&MatchKind.Sequenced)!=0):
                        sectionContext.subParserIndex=idxAtt
                    return (parseRes,newCtx)
        if (self.kind&MatchKind.Weak)!=0 or (self.kind&matchKind)==0:
            return (ParserControl.MaybeMatch,None)
        if ((self.kind&(MatchKind.Repeating|MatchKind.AlwaysActive|MatchKind.Floating)) != 0
                or ((self.kind&MatchKind.Repeatable)!=0 and (self.kind&matchKind&MatchKind.Fallback)!=0)
                or subIndex is None):
            (parseRes,sectionContext)=sectionContext.parse(parser=parser, mode=ParseMode.Speculative)
            nLog.debug("speculative rematch of %s yields %s",self.name,parseRes)
            if parseRes in [ParserControl.Open, ParserControl.Closed, ParserControl.CloseNormal, ParserControl.CloseForced]:
                return (parseRes,sectionContext)
        nLog.debug("rematch of %s failed %s %s",self.name,self.kind,matchKind)
        return (ParserControl.Failed,None)

    def close(self,parser,sectionContext, closeKind):
        """called after parsing all subParsers or
        when implicitly closing the section"""
        if len(self.subParsers)==0 or sectionContext.subParserIndex == len(self.subParsers):
            return closeKind
        return CloseKind.Forced

class SLParser(LineParser):
    """simple line parser that has just a single regular expression"""
    def __init__(self,startReStr,name=None,**kwds):
        if name is None:
            name="SLParser-"+repr(startReStr)[1:-1]
        super(SLParser,self).__init__(name,
            startReStr=startReStr,**kwds)

class FixedValueParser(LineParser):
    """a simple parser that when triggered sets a predefined value to a given key"""
    def __init__(self,startReStr, matchDict, name=None,**kwds):
        if name is None:
            name="FixedValParser-"+repr(startReStr)[1:-1]
        super(FixedValueParser,self).__init__(name,
            startReStr=startReStr,**kwds)
        self.matchDict=matchDict
    def handleMatch(self, matchDict, parser, sectionContext):
        matchDict.update(self.matchDict)
        return super(FixedValueParser,self).handleMatch(matchDict,parser,sectionContext)

class SkipSome(LineParser):
    """simple parser that skips some lines"""
    def __init__(self):
        super(SkipSome,self).__init__("SkipSome",kind=MatchKind.Weak|MatchKind.Sequenced|MatchKind.Repeating)

    def parse(self,parser,sectionContext,mode):
        line=parser.lineFile.line
        if not line:
            return (ParserControl.AtEnd,sectionContext)
        if mode==ParseMode.Confirmed or ((line=="" or line.isspace()) and mode != ParseMode.Speculative):
            return (ParserControl.Closed,sectionContext)
        return (ParserControl.MaybeMatch,sectionContext)

class RootSection(LineParser):
    def open(self,parser,sectionContext,mode):
            parser.context.newRun()
    def close(self,parser,sectionContext, closeKind):
        parser.context.endRun(closeKind)
        return super(RootSection,self).close(parser,sectionContext,closeKind)

class SectionOpener(LineParser):
    def __init__(self,name,sectionProperty=None,**kwds):
        super(SectionOpener,self).__init__(name,**kwds)
    def open(self,parser,sectionContext,mode):
        p=self.prop(sectionContext)
        if p is None:
            parser.logger.addWarning("SectionOpener could not find property {0}".format(self.name))
        p.handleProperty(parser.logger,globalContext=parser, localContext=sectionContext, value={})

class CollectDict(LineParser):
    def __init__(self,name, startReStr, kind=MatchKind.Sequenced|MatchKind.Repeating, nItems=None, valProperty=None, **kwds):
        super(CollectDict,self).__init__(name=name,startReStr=startReStr,kind=kind,**kwds)
        self.nItems=nItems
        self.valProperty=valProperty

    def strToVal(self,val,logger):
        if not self.valProperty is None:
            return self.valProperty.strToVal(val,logger)
        return val
    def parse(self, parser, sectionContext, mode):
        if not self.nItems is None and sectionContext.context.get('nItems',0) > self.nItems:
            return (ParserControl.Failed,sectionContext)
        return super(CollectDict,self).parse(parser=parser, sectionContext=sectionContext, mode=mode)

    def handleMatch(self, matchDict, parser, sectionContext):
        ctx=sectionContext.context
        ctx['nItems']=ctx.get('nItems',0)+1
        resDict=ctx.get('resDict',{})
        key=matchDict.get("key","")
        value=matchDict.get("value","")
        if key in resDict:
            parser.logger.addWarning("CollectDict {0} had duplicate key {1}".format(self.name, key))
        resDict[key]=self.strToVal(value,parser.logger)
        ctx['resDict']=resDict
        return True

    def close(self,parser,sectionContext,closeKind):
        if closeKind==CloseKind.Forced:
            if "resDict" in sectionContext.context: # else store an empty dict??
                p=Property.withName(self.name)
                if p is None:
                    parser.logger.addWarning("LineParser {0} has no corresponding property, ignoring parsed value {1}".format(self.name,sectionContext.context["resDict"]))
                else:
                    p.handleProperty(logger=parser.logger,globalContext=parser, localContext=sectionContext, value=sectionContext.context["resDict"])
        return super(CollectDict,self).close(parser=parser,
            sectionContext=sectionContext,closeKind=closeKind)

def propForUnknownControlInKey(key):
    propName="controlInUnknown_"+key.lower()
    prop=Property.withName(propName)
    if prop is None:
        prop=Prop(propName,repeats=True,
        description="Auto generated property for unknown key {0}".format(key),
        superKeys=["UnknownControlInKeys"])
        #properties.append(prop)
    return prop

def propForUnknownGeometryInKey(key):
    propName="geometryInUnknown_"+key.lower()
    prop=Property.withName(propName)
    if prop is None:
        prop=Prop(propName,repeats=True,
        description="Auto generated property for unknown key {0}".format(key),
        superKeys=["UnknownGeometryInKeys"])
        #properties.append(prop)
    return prop

def propForUnhandledSpeciesKey(key):
    if not key in ["basis_acc", "prodbas_acc", "innermost_max", "cut_pot",
        "cutoff_type", "cut_free_atom", "cut_core", "basis_dep_cutoff",
        "cut_atomic_basis", "radial_base", "radial_multiplier", "angular",
        "angular_min", "angular_acc", "angular_grids", "logarithmic", "l_hartree",
        "core_states", "KH_core_states", "include_min_basis", "max_n_prodbas",
        "max_l_prodbas", "pp_charge", "pp_local_component"]:
        return None
    propName="species_"+key.lower()
    prop=Property.withName(propName)
    if prop is None:
        prop=Prop(propName,repeats=True,
        description="Auto generated property for unhandled species key {0}".format(key),
        superKeys=["ParticleKindParameters","UnknownControlInKeys"])
        #properties.append(prop)
    return prop

class UnknownKey(LineParser):
    def __init__(self,name,startReStr,key2Prop, kind=MatchKind.Fallback|MatchKind.Repeatable, **kwds):
        super(UnknownKey,self).__init__(name=name,startReStr=startReStr,kind=kind,**kwds)
        self.key2Prop=key2Prop

    def handleMatch(self, matchDict, parser, sectionContext):
        """tries to parse a non defined extra key of control in"""
        key=matchDict["key"]
        value=matchDict.get("value","")
        if value is None:
            value=""
        else:
            value=value.strip()
        prop=self.key2Prop(key)
        if prop is None:
            return False
        prop.handleProperty(logger=parser.logger,globalContext=parser, localContext=sectionContext, value=value)
        return True

class GroupSection(LineParser):
    """A parser used only to group other parsers"""
    def __init__(self,name,subParsers=None, **kwds):
        super(GroupSection,self).__init__(name, startReStr=None,
                subParsers=subParsers,**kwds)

    def parse(self,parser,sectionContext,mode):
        (parseRes, subSection)=self.rematch(parser,sectionContext,sectionContext,matchKind=MatchKind.Sequenced|MatchKind.AlwaysActive)
        if (subSection is None):
            (parseRes,subSection)=self.rematch(parser,sectionContext,sectionContext,matchKind=MatchKind.Fallback)
        if (subSection is None):
            return (ParserControl.Failed,sectionContext)
        return (parseRes,subSection)

def storeConfigProps(parser,ctx):
    "recursively stores the properties in ctx as configuration properties"
    for k,v in ctx.items():
        try:
            prop=Property.withName(k)
            if prop is None:
                parser.logger.addWarning("Could not find property for key {0} in storeConfigProps".format(k))
            elif prop.kind == PropertyKind.commonProperty or prop.kind == PropertyKind.configProperty:
                if not prop.dtype is None:
                    prop.addConfigProperty(parser.logger, parser, v)
            elif prop.kind == "section":
                if prop.repeats:
                    for el in v:
                        storeConfigProps(parser,el)
                else:
                    storeConfigProps(parser,v)
            else:
                parser.logger.addWarning("Unexpected property kind {0} of property {1} in storeConfigProps".format(prop.kind,prop.name))
        except:
            raise Exception("Exception while handling property {0} from {1} had exception {2}".format(k, ctx, traceback.format_exc()))

def collectScfConfigProp(parser,iterData, iIter,nIterTarget, targetCtx):
    "recursively stores the properties in the iteration iterData into targetCtx"
    for k,v in iterData.items():
        prop=Property.withName(k)
        if prop is None:
            parser.logger.addWarning("Could not find property for key {0} in collectScfConfigProp".format(k))
        elif prop.kind == PropertyKind.commonProperty or prop.kind == PropertyKind.configProperty:
            if prop.dtype is None:
                continue
            shape=numpy.shape(v)
            newShape=list(nIterTarget)+list(shape)
            if not prop.name in targetCtx:
                targetCtx[prop.name]=numpy.zeros(tuple(newShape),dtype=prop.dtype)
                targetCtx[prop.name][iIter]=v
            elif targetCtx[prop.name].shape != tuple(newShape):
                parser.logger.addWarning("Non uniform shape in property {0} ({1}), {2} vs {3}, not supported, skipping".format(k, iIter, targetCtx[prop.name].shape, newShapeq))
            else:
                targetCtx[prop.name][iIter]=v
        elif prop.kind == "section":
            if prop.repeats:
                ii=0
                myNIterTarget=list(nIterTarget)
                myNIterTarget.append(len(v))
                myIIter=list(iIter)
                myIIter.append(ii)
                for el in v:
                    collectScfConfigProp(parser,el,tuple(myIIter),tuple(myNIterTarget),targetCtx)
                    ii+=1
                    myIIter[-1]=ii
            else:
                collectScfConfigProp(parser,v,iIter,nIterTarget,targetCtx)
        else:
            parser.logger.addWarning("Unexpected property kind {0} of property {1} in collectScfConfigProp".format(prop.kind,prop.name))

class BandParser(SectionOpener):
    "Parses a band output"
    def storeProps(self,parser,sectionContext):
        #print 'readFile1'
        for expectedValue in ['startX','startY','startZ','endX','endY','endZ','nPoints']:
            if not (expectedValue in sectionContext.context):
                parser.logger.addWarning("missing value for {0} in Band {1}".format(expectedValue, sectionContext.context['bandNr']))
                return
        ctxs=sectionContext.context["ConfigurationProperties"]["ScfInfo"]
        if len(ctxs)!=1:
            parser.logger.addWarning("Unexpected number of scf procedures in ScfInfo")
        #print 'readFile'

    def close(self,parser,sectionContext,closeKind):
        self.storeProps(parser,sectionContext)
        sectionContext.context.clear()
        return super(BandParser,self).close(parser=parser,
            sectionContext=sectionContext,closeKind=closeKind)

class ScfParser(SectionOpener):
    "Parses an scf cycle"
    def storeProps(self,parser,sectionContext):
        if not ("ConfigurationProperties" in sectionContext.context and "ScfInfo" in sectionContext.context["ConfigurationProperties"]):
            return
        ctxs=sectionContext.context["ConfigurationProperties"]["ScfInfo"]
        if len(ctxs)!=1:
            parser.logger.addWarning("Unexpected number of scf procedures in ScfInfo")
        for ctx in ctxs:
            d2=dict(ctx)
            scfIters=[]
            if "ScfIteration" in d2:
                scfIters=d2["ScfIteration"]
                del d2["ScfIteration"]
            storeConfigProps(parser,d2)
            if scfIters:
                nIters=len(scfIters)
                if nIters>0 and scfIters[-1].get("scfIterationNr",-1)!=nIters:
                    parser.logger.addWarning("Mismatch between parsed scf iterations {0} and interation number of the last iteration {1}".format(nIters,scfIters[-1].get("scfIterationNr",-1)))
                iterData={"nScfDft":nIters}
                nIterToKeep=10
                if nIters>nIterToKeep:
                    scfIters=scfIters[nIters-nIterToKeep:]
                for i in range(len(scfIters)):
                    collectScfConfigProp(parser,scfIters[i],(nIterToKeep-len(scfIters)+i,),
                            (nIterToKeep,),iterData)
                storeConfigProps(parser,iterData)

    def close(self,parser,sectionContext,closeKind):
        self.storeProps(parser,sectionContext)
        sectionContext.context.clear()
        return super(ScfParser,self).close(parser=parser,
            sectionContext=sectionContext,closeKind=closeKind)

class ConfigParser(SectionOpener):
    "Parses a configuration"
    def storeConfig(self, parser, sectionContext):
        if not ("ConfigurationProperties" in sectionContext.context and "CoreConfiguration" in sectionContext.context["ConfigurationProperties"]):
            return
        ctx=sectionContext.context["ConfigurationProperties"]["CoreConfiguration"]
        parser.context.configIndex+=1
        particles=ctx['particlePos']
        pPos=numpy.zeros((len(particles),3),dtype=float)
        pNames=[]
        iParticle=0
        for p in particles:
            if p['iParticle'] != iParticle + 1:
                parser.logger.addWarning("iParticle mismatch parsing configuration, {0} vs {1}".format(p['iParticle'],iParticle))
            pPos[iParticle]=[p['x'],p['y'],p['z']]
            pNames.append(p['kindName'])
            iParticle += 1
        config={
        "particlePositions":pPos,
        "particleKindNames":pNames
        }
        storeConfigProps(parser,config)

        #periodic = maybeGet(parser.context.runInfo, ["SimulationParameters", "GeometryInValues", "geometryIn_lattice_vector"])
        #if periodic:
        #    lattice = [[periodic[i]['x'], periodic[i]['y'], periodic[i]['z']] for i in range(3)]
        #    pbc = True
        #else:
        #    lattice = [[parser.rootParser.tilde_obj.PERIODIC_LIMIT, 0, 0], [0, parser.rootParser.tilde_obj.PERIODIC_LIMIT, 0], [0, 0, parser.rootParser.tilde_obj.PERIODIC_LIMIT]]
        #    pbc = False
        #
        #parser.rootParser.tilde_obj.structures.append(  Atoms(symbols=pNames, cell=lattice, positions=pPos, pbc=pbc)  )

    def close(self, parser, sectionContext, closeKind):
        self.storeConfig(parser, sectionContext)
        sectionContext.context.clear()
        return super(ConfigParser, self).close(parser=parser, sectionContext=sectionContext, closeKind=closeKind)

class GaussianParser(LineParser):
    """Parser for contracted gaussian functions

    startReStr is supposed to define the l_shell, n_contracted and (optionally) alpha"""
    def parse(self, parser, sectionContext, mode):
        m=self.startRe.match(parser.lineFile.line)
        if not m:
            return (ParserControl.Failed,sectionContext)
        n_contracted=int(m.group("n_contracted"))
        gauss=[]
        if n_contracted==1:
            gauss.append({"alpha":float(m.group("alpha")), "coeff":1})
        else:
            for i in range(n_contracted):
                line=lineFile.readline()
                line.split()
                gauss.append({"alpha":float(line[0]), "coeff":float(line[1])})
        self.prop(sectionContext).handleProperty(logger=parser.logger, globalContext=parser,localContext=sectionContext, value=gauss)
        return (ParserControl.Closed,sectionContext)

class ValueHandlingParser(LineParser):
    """Parser that treats specially the re group 'value'"""
    def __init__(self,name,startReStr,valueHandler,**kwds):
        super(ValueHandlingParser,self).__init__(name=name,startReStr=startReStr,**kwds)
        self.valueHandler=valueHandler

    def handleMatch(self, matchDict, parser, sectionContext):
        return self.valueHandler(parser.logger,parser,sectionContext,matchDict["value"])

class DictGroupParser(LineParser):
    """puts the groupdict of startRe into a property given by the name of this.
    Single values might be converted if type is given in valueKinds"""
    def __init__(self, name, valueKinds=None, **kwds):
        super(DictGroupParser,self).__init__(name,**kwds)
        if (valueKinds is None):
            self.valueKinds={}
        else:
            self.valueKinds=valueKinds

    def handleMatch(self,matchDict,parser,sectionContext):
        matchDict=dict(matchDict)
        for k,v in self.valueKinds.items():
            if k in matchDict:
                matchDict[k]=Property.strToValKind(v,matchDict[k],parser.logger,contextName=self.name)
        p=None
        if not self.propNameMapper is None:
            p=Property.withName(self.propNameMapper(self.name))
        if p is None:
            p=Property.withName(self.name)
        if p is None:
            parser.logger.addWarning("Could not find property named {0}".format(self.name))
        else:
            p.handleProperty(logger=parser.logger,globalContext=parser, localContext=sectionContext, value=matchDict)
        return True

class OneSub(SectionOpener):
    """parses one of the subParsers"""
    def parse(self, parser, sectionContext, mode):
        parseRes,ctx=super(OneSub,self).parse(parser,sectionContext,mode)
        lastSuccessfulParse=None
        if not parseRes in [ParserControl.Open]:
            return (parseRes, ctx)
        while True:
            ctx=parser.handleMatch(ctx,parseRes) # TODO by EB
            if ctx == sectionContext:
                break
            subCtx=ctx
            while not subCtx is None and subCtx.superSection != sectionContext:
                subCtx=subCtx.superSection
            if (subCtx is None or subCtx.superSection != sectionContext or
                (not lastSuccessfulParse is None and subCtx.section != lastSuccessfulParse)):
                break
            oldCtx=ctx
            parseRes,ctx=subCtx.parse(parser,ctx,mode)
            if oldCtx.section in self.subParsers:
                if parseRes == ParserControl.Open:
                    lastSuccessfulParse = oldCtx.section
                    continue
                elif parseRes in [ParserControl.Closed,ParserControl.CloseNormal]:
                    break
                elif parseRes == ParserControl.CloseForced:
                    if not lastSuccessfulParse is None:
                        break
            if ctx == sectionContext:
                break
            subCtx=ctx
            while not subCtx is None and subCtx.superSection != sectionContext:
                subCtx=subCtx.superSection
            if (subCtx is None or subCtx.superSection != sectionContext or
                (not lastSuccessfulParse is None and subCtx.section != lastSuccessfulParse)):
                break
        if ctx == sectionContext:
            if lastSuccessfulParse is None:
                return (ParserControl.CloseForced, ctx)
            else:
                return (ParserControl.CloseNormal,ctx)
        subCtx=ctx
        while not subCtx is None and subCtx.superSection != sectionContext:
            subCtx=subCtx.superSection
        if (subCtx is None or subCtx.superSection != sectionContext):
            return (parseRes,ctx)
        if lastSuccessfulParse is None:
            assert False, "unexpected state"
        if subCtx.section == lastSuccessfulParse:
            parser.closeSectionsUpTo(ctx,CloseKind.Forced,sectionContext)
            return (ParserControl.CloseNormal,sectionContext)
        assert False, "unexpected state"

class SubFileParser(LineParser):
    """A parser of a file that might be inline or external"""
    def __init__(self,name,firstSubFileParser,preHeader,postHeader,extFilePath,**kwds):
        super(SubFileParser,self).__init__(name=name,**kwds)
        self.firstSubFileParser=firstSubFileParser
        self.preHeader=preHeader
        self.postHeader=postHeader
        self.extFilePath=extFilePath

    def parse(self, parser, sectionContext, mode):
        m=self.startRe.match(parser.lineFile.line)
        if not m:
            return (ParserControl.Failed,sectionContext)
        notInline=False
        if "notInline" in m.groupdict() and m.group("notInline"):
            notInline=True
        subLineFile=self.subLineFile(parser.lineFile,parser,sectionContext,not notInline)
        if subLineFile is None:
            return (ParserControl.CloseForced,sectionContext)
        subParser=FileParser(lineFile=subLineFile,context=parser.context,
                firstParserContext=SectionContext(self.firstSubFileParser), rootParser=parser)
        sectionContext.context["subParser"]=subParser
        subParser.parse()
        nLog.debug("finished subParser %s %s",self.name,parser.lineFile.posStr())
        if sectionContext.context["parseHeader"]:
            for headerRe in self.postHeader[1:]:
                oldLine=lineFile.line
                if not re.match(headerRe,lineFile.readline()):
                    parser.logger.addWarning("{0} failed to match post-header {1}".format(self.name,headerRe))
                    lineFile.pushBack(oldLine)
                    break
        return (ParserControl.Closed, sectionContext)

    def subLineFile(self, lineFile, parser, sectionContext, parseHeader):
        controlInLineFile=None
        if parseHeader:
            for headerRe in self.preHeader:
                if not re.match(headerRe,lineFile.readline()):
                    parser.logger.addWarning("failed to match header for {0} against {1}".format(self.name,headerRe))
                    lines=[]
                    separatorRe=re.compile(self.preHeader[-1])
                    for i in range(5):
                        lines.append(lineFile.readline())
                        if separatorRe.match(lines[-1]):
                            break
                    if not separatorRe.match(lines[-1]):
                        lines.reverse()
                        lineFile.pushBack(lines)
                        parser.logger.addWarning("failed to find separator {0} giving up in-line parsing of {1}, trying to fall back to external file".format(separatorRe,self.name))
                        parseHeader=False
                    break
            if len(self.postHeader)==0:
                parseHeader=False
            if parseHeader:
                while True:
                    line=lineFile.readline()
                    if not line or not line.isspace():
                        break
                separatorRe=re.compile(self.postHeader[0])
                if separatorRe.match(lineFile.line):
                    parser.logger.addNote("failed to find inline file in {0}, falling back to external file".format(self.name))
                else:
                    nLog.debug("setting up inline parser for %s",self.name)
                    controlInLineFile=SubLineFile(lineFile,separatorRe)
                    controlInLineFile.pushBack([" "])
        parsedExternalFile=False
        if controlInLineFile is None:
            parsedExternalFile=True
            extFilePath=os.path.normpath(os.path.join(os.path.dirname(parser.context.path),self.extFilePath))
            if not os.path.exists(extFilePath):
                parser.logger.addWarning("Skipping parsing of non existing externalFile {0} triggered by parsing of {1}".format(extFilePath,self.name))
                return None
            parser.context.runTrajInfo['derived_from'].add(parser.context.normalizer.sha224OfPath(extFilePath))
            normalizer=parser.context.normalizer
            cInMeta=normalizer.metaOfPath(extFilePath)
            outMeta=normalizer.metaOfPath(parser.context.path)
            if "mtime" in cInMeta and "ctime" in outMeta and "mtime" in outMeta:
                if cInMeta["mtime"]>outMeta["ctime"] and cInMeta["mtime"]>outMeta["mtime"]:
                    parser.logger.addWarning("{!r} modification date seems to be *after* the creation or modification time of the output, thus data in {!r} might not correspond to this run {!r}.".format(extFilePath,parser.context.path))
            else:
                parser.logger.addWarning("Missing complete date meta info for {!r} and/or {!r}.".format(extFilePath,parser.context.path))
            kwds=dict()
            if sys.version_info.major>2:
                kwds["encoding"]=normalizer.metaOfPath(extFilePath).get("encoding","utf_8")
            controlInLineFile=LineFile(open(extFilePath,**kwds))
        sectionContext.context["subLineFile"]=controlInLineFile
        sectionContext.context["parseHeader"]=parseHeader
        return controlInLineFile

class LineFile(object):
    """a simple file wrapper that keeps track of the line number"""

    def __init__(self,textFile, **kwds):
        super(LineFile,self).__init__(**kwds)
        self.nextLines=[]
        self.file=textFile
        self.lineNr=0
        self.line=""

    def readline(self):
        if self.nextLines:
            self.line=self.nextLines.pop()
            assert self.line, "empty line in LineFile.nextLines"
        else:
            self.line=self.file.readline()
        if self.line:
            self.lineNr+=1
        return self.line

    def pushBack(self,lines):
        if self.line:
            self.nextLines.append(self.line)
        else:
            self.lineNr += 1
        self.nextLines += lines
        self.lineNr-= len(lines)
        return self.readline()

    def posDict(self,**kwds):
        res={
            'file':self.file.name,
            'lineNr':self.lineNr,
            'line':self.line
        }
        res.update(kwds)
        return res

    def posStr(self):
        return "{0}:{1} ({2})".format(self.file.name, self.lineNr, repr(self.line)[1:-1])

class SubLineFile(object):
    def __init__(self,subLineFile,endRe):
        self.endRe=endRe
        self.subLineFile=subLineFile
        self.line=self.subLineFile.line

    def readline(self):
        if not self.endRe.match(self.subLineFile.line):
            self.line=self.subLineFile.readline()
            if self.endRe.match(self.line):
                self.line=""
        return self.line

    def pushBack(self,lines):
        self.line=self.subLineFile.pushBack(lines)

    def posDict(self,**kwds):
        return self.subLineFile.posDict(**kwds)

    def posStr(self):
        return self.subLineFile.posStr()

def dumpParsers(parser,indent="",printed=set()):
    """Print out the hierarchical structure of the parsers"""
    if parser in printed:
        nLog.warn("%s%s ***Warning*** repeated, not recursing",indent,parser.name)
        return
    printed.add(parser)
    nLog.info(indent+parser.name)
    for subP in parser.subParsers:
        dumpParsers(subP,indent+"    ",printed)

class SectionContext(object):
    """Keeps the context of a level of the parsing stack.

    Call the close and parse methods of the contxt rather than the section directly."""
    def __init__(self,section,subParserIndex=-1,context=None,
                superSection=None, tmpContext=False, **kwds):
        super(SectionContext,self).__init__(**kwds)
        self.didFail=0
        self.section=section
        self.subParserIndex=subParserIndex
        self.context=context
        self.matched=set()
        self.superSection=superSection
        self.tmpContext=tmpContext
        if self.context is None:
            self.context={}

    def close(self,parser,closeKind):
        if (closeKind==CloseKind.Forced and self.didFail>0 and (self.section.kind&MatchKind.ExpectFail) == 0 and
                not (parser.lineFile.line == "" or parser.lineFile.line.isspace())):
            parser.logger.addWarning("unexpected parsing failure of section {0}".format(self.section.name))
        for sect in self.section.subParsers:
            if (sect.kind&MatchKind.Required)!=0 and not sect in self.matched:
                parser.logger.addWarning("missing required section {0} of {1}".format(sect.name,self.section.name))
        self.matched.clear()
        return self.section.close(parser=parser,sectionContext=self,closeKind=closeKind)

    def rematch(self,parser,lastSection, matchKind):
        return self.section.rematch(parser=parser,sectionContext=self,lastSection=lastSection, matchKind=matchKind)

    def parse(self,parser,mode):
        xxx=self.section.parse(parser=parser,sectionContext=self,mode=mode)
        (parseRes,newContext)=xxx
        if parseRes in [ParserControl.Failed]:
            if mode in [ParseMode.Normal, ParseMode.Confirmed] and self.didFail==0:
                self.didFail+=1
        elif (parseRes in [ParserControl.Open, ParserControl.Closed, ParserControl.CloseNormal] or
            (parseRes == ParserControl.MaybeMatch and mode == ParseMode.Confirmed)):
            self.didFail-=1
            if not self.superSection is None:
                self.superSection.matched.add(self.section)
        else:
            assert parseRes in [ParserControl.CloseForced,ParserControl.MaybeMatch]
        return (parseRes,newContext)

def stackNames(stack):
    res=[]
    while not stack is None:
        res.append(stack.section.name)
        stack=stack.superSection
    return res

class FileParser(object):
    """A parser for a file, this contains the machinery to parse
    LineParsers and recover from failures to match.
    The result of the parsing is expected to go into the context"""

    def addMessage(self,message,level=0):
        self.context.logger.addMessage(message+" at "+self.lineFile.posStr())

    def __init__(self, lineFile, context, firstParserContext, repeatsFirst=False, rootParser=None):
        self.sectionStack=firstParserContext
        self.sectionIndex=[]
        self.repeatsFirst=repeatsFirst
        self.lineFile=lineFile
        self.context=context
        self.logger=Logger(self.addMessage)
        self.rootParser=rootParser

    def goToNextSection(self, lastSect, parseRes, repeatsFirst=False):
        "goes to the next section to parse, and returns the updated stack"
        nLog.debug("goToNextSection(%r, %r %r)",stackNames(lastSect),parseRes,repeatsFirst)
        if (not parseRes in [ParserControl.Failed, ParserControl.AtEnd]) and (lastSect.section.kind & MatchKind.Repeating)!=0:
            return lastSect
        closeMode=CloseKind.Normal
        if not parseRes in [ParserControl.Closed, ParserControl.CloseNormal, ParserControl.Open]:
            closeMode=CloseKind.Forced
        while lastSect.superSection:
            parentSect=lastSect.superSection
            nSubParsers=len(parentSect.section.subParsers)
            hasActiveParsers=False
            for i in range(1,nSubParsers+1):
                iSection=parentSect.subParserIndex+i
                if iSection>=nSubParsers and (parentSect.section.kind&MatchKind.LastNotEnd)==0:
                    break
                iSection=iSection%nSubParsers
                nextParser=parentSect.section.subParsers[iSection]
                hasActiveParsers=hasActiveParsers or (nextParser.kind&(MatchKind.Repeatable|MatchKind.Repeating))!=0 or not nextParser in parentSect.matched
                if (nextParser.kind & MatchKind.Sequenced) != 0 and not ((nextParser.kind&(MatchKind.Repeatable|MatchKind.Repeating))==0 and nextParser in parentSect.matched):
                    parentSect.subParserIndex=iSection
                    return SectionContext(nextParser, superSection=parentSect)
            if ((parentSect.section.kind&MatchKind.LastNotEnd)!=0):
                return parentSect
            closeMode=parentSect.close(parser=self,closeKind=closeMode)
            lastSect=parentSect
            if parseRes != ParserControl.AtEnd and parentSect.section.kind&(MatchKind.Repeating|MatchKind.AlwaysActive|MatchKind.Floating)!=0:
                return parentSect
        if lastSect and repeatsFirst:
            return lastSect
        return None

    def closeSectionsUpTo(self, lastSect, mode, baseSection):
        origStack=lastSect
        while (not lastSect is baseSection and not lastSect is None):
            mode=lastSect.close(parser=self,closeKind=mode)
            lastSect=lastSect.superSection
        if not baseSection is None and not lastSect is baseSection:
            self.logger.addWarning("closeSectionsUpTo baseSection {0} not found in {1}".format(baseSection.section.name,stackNames(origStack)))
        return lastSect

    def resyncParsing(self, lastSect, matchKind):
        """checks various possible matches to resynchronize the parsing.

        Returns true if synchronization was successful"""
        nLog.debug("*** resyncParsing %r %r %r at %s",stackNames(self.sectionStack),stackNames(lastSect),matchKind,
                self.lineFile.posStr())
        if self.lineFile.line == "" or self.lineFile.line.isspace():
            return None
        sectAtt=lastSect.superSection
        iSection=0
        while not sectAtt is None:
            iSection+=1
            (parseRes,sect)=sectAtt.rematch(parser=self, lastSection=lastSect, matchKind=matchKind)
            if sect is None:
                sectAtt=sectAtt.superSection
                continue
            if nLog.isEnabledFor(logging.DEBUG):
                msg="performed resync from section {0} ({1}) of {2} to {3}".format(sectAtt.section.name, iSection, stackNames(lastSect), stackNames(sect))
                nLog.debug("%s %s",msg,self.lineFile.posStr())
            if sect is sectAtt or (sect.section.kind&MatchKind.Floating)==0:
                lastSect=self.closeSectionsUpTo(lastSect=lastSect, mode=CloseKind.Forced,baseSection=sectAtt)
            assert parseRes in [ParserControl.Open, ParserControl.Closed, ParserControl.CloseNormal, ParserControl.CloseForced]
            nLog.debug("*** resyncSec successful return")
            return self.handleParsing(sect,parseRes)
        nLog.debug("*** resyncSec failure return")
        return None

    def parseStep(self, lastSect):
        line=self.lineFile.readline()
        nLog.debug("\n** Start parsing of %s",self.lineFile.posStr())
        if not line:
            self.closeSectionsUpTo(lastSect=lastSect,mode=CloseKind.Forced,baseSection=None)
            return None
        (parseRes,sectionContext)= lastSect.parse(parser=self,mode=ParseMode.Normal)
        if nLog.isEnabledFor(logging.DEBUG) and not self.sectionStack is sectionContext:
            nLog.debug("stack after parsing %r -> %r", stackNames(self.sectionStack),stackNames(sectionContext))
        sectionStack=self.handleParsing(sectionContext,parseRes)
        nLog.debug("stack after handling %r", stackNames(self.sectionStack))
        return sectionStack

    def parse(self):
        while self.sectionStack:
            self.sectionStack = self.parseStep(self.sectionStack)

    def handleParsing(self,lastSect,parseRes):
        nLog.debug("handling parsing of %s with result %r",lastSect.section.name,parseRes)
        if parseRes == ParserControl.Open:
            if len(lastSect.section.subParsers)==0:
                logger.addWarning("section {0} returned ParserControl.Open without subParsers, closing".format(lastSect.section.name))
                lastSect.close(parser=self, closeKind=CloseKind.Normal)
                return self.goToNextSection(lastSect,parseRes, repeatsFirst=self.repeatsFirst)
            lastSect.subParserIndex=0
            return SectionContext(lastSect.section.subParsers[0],
                    superSection=lastSect)
        elif parseRes == ParserControl.Closed:
            return self.goToNextSection(lastSect,parseRes)
        elif parseRes == ParserControl.CloseNormal:
            lastSect.close(parser=self,closeKind=CloseKind.Normal)
            return self.goToNextSection(lastSect,parseRes,repeatsFirst=self.repeatsFirst)
        elif parseRes == ParserControl.CloseForced:
            lastSect.close(parser=self,closeKind=CloseKind.Forced)
            return self.goToNextSection(lastSect,parseRes,repeatsFirst=self.repeatsFirst)
        elif parseRes == ParserControl.MaybeMatch:
            resyncSec = self.resyncParsing(lastSect=lastSect,matchKind=MatchKind.Sequenced|MatchKind.AlwaysActive)
            if resyncSec:
                return resyncSec
            resyncSec=self.resyncParsing(lastSect=lastSect,matchKind=MatchKind.Fallback)
            if resyncSec:
                return resyncSec
            (parseRes,newSection)=lastSect.parse(parser=self,mode=ParseMode.Confirmed)
            if parseRes is ParserControl.MaybeMatch:
                if lastSect.subParsers:
                    pareseRes=ParserControl.Open
                else:
                    lastSect.close(parser=self,closeKind=CloseKind.Normal)
                    parseRes=ParserControl.Closed
            return self.handleParsing(newSection,parseRes)
        elif parseRes == ParserControl.Failed:
            if ((lastSect.section.kind&(MatchKind.AlwaysActive|MatchKind.Floating))!=0
                    and (lastSect.section.kind&MatchKind.Sequenced)==0) :
                lastSect.close(parser=self, closeKind=CloseKind.Forced)
                return self.goToNextSection(lastSect,parseRes,repeatsFirst=self.repeatsFirst)
            resyncSec=self.resyncParsing(lastSect,matchKind=MatchKind.Sequenced|MatchKind.AlwaysActive)
            if resyncSec:
                return resyncSec
            resyncSec=self.resyncParsing(lastSect,matchKind=MatchKind.Fallback)
            if resyncSec:
                return resyncSec
            return lastSect
        elif parseRes == ParserControl.AtEnd:
            return self.closeSectionsUpTo(lastSect,parseRes,None)
        else:
            assert False,"unexpected parseRes state {0} in handleParsing".format(parseRes)
