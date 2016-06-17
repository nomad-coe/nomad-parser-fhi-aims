package eu.nomad_lab.parsers

import eu.nomad_lab
import eu.nomad_lab.DefaultPythonInterpreter
import org.{ json4s => jn }
import eu.{ nomad_lab => lab }
import scala.collection.breakOut

object FhiAimsParser extends SimpleExternalParserGenerator(
  name = "FhiAimsParser",
  parserInfo = jn.JObject(
    ("name" -> jn.JString("FhiAimsParser")) ::
      ("parserId" -> jn.JString("FhiAimsParser" + lab.FhiAimsVersionInfo.version)) ::
      ("versionInfo" -> jn.JObject(
        ("nomadCoreVersionInfo" -> jn.JObject(lab.NomadCoreVersionInfo.toMap.map {
          case (k, v) => k -> jn.JString(v.toString)
        }(breakOut): List[(String, jn.JString)])) ::
          (lab.FhiAimsVersionInfo.toMap.map {
            case (key, value) =>
              (key -> jn.JString(value.toString))
          }(breakOut): List[(String, jn.JString)])
      )) :: Nil
  ),
  mainFileTypes = Seq("text/.*"),
  mainFileRe = """\s*Invoking FHI-aims \.\.\.
\s*Version """.r,
  cmd = Seq(DefaultPythonInterpreter.python2Exe(), "${envDir}/parsers/fhi-aims/parser/parser-fhi-aims/FhiAimsParser.py",
    "--uri", "${mainFileUri}", "${mainFilePath}"),
  resList = Seq(
    "parser-fhi-aims/FhiAimsParser.py",
    "parser-fhi-aims/FhiAimsCommon.py",
    "parser-fhi-aims/FhiAimsControlInParser.py",
    "parser-fhi-aims/FhiAimsBandParser.py",
    "parser-fhi-aims/FhiAimsDosParser.py",
    "parser-fhi-aims/setup_paths.py",
    "nomad_meta_info/public.nomadmetainfo.json",
    "nomad_meta_info/common.nomadmetainfo.json",
    "nomad_meta_info/meta_types.nomadmetainfo.json",
    "nomad_meta_info/fhi_aims.nomadmetainfo.json"
  ) ++ DefaultPythonInterpreter.commonFiles(),
  dirMap = Map(
    "parser-fhi-aims" -> "parsers/fhi-aims/parser/parser-fhi-aims",
    "nomad_meta_info" -> "nomad-meta-info/meta_info/nomad_meta_info",
    "python" -> "python-common/common/python/nomadcore"
  ) ++ DefaultPythonInterpreter.commonDirMapping(),
  metaInfoEnv = Some(lab.meta.KnownMetaInfoEnvs.fhiAims)
)
