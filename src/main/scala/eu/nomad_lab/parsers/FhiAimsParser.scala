package eu.nomad_lab.parsers
import eu.nomad_lab.DefaultPythonInterpreter
import org.{json4s => jn}

object FhiAimsParser extends SimpleExternalParserGenerator(
      name = "FhiAimsParser",
      parserInfo = jn.JObject(
        ("name" -> jn.JString("FhiAimsParser")) ::
          ("version" -> jn.JString("1.0")) :: Nil),
      mainFileTypes = Seq("text/.*"),
      mainFileRe = """\s*Invoking FHI-aims \.\.\.
\s*Version """.r,
      cmd = Seq(DefaultPythonInterpreter.python2Exe(), "${envDir}/parsers/fhi-aims/parser/parser-fhi-aims/SimpleFhiParser.py",
        "--uri", "${mainFileUri}", "${mainFilePath}"),
      resList = Seq(
        "parser-fhi-aims/SimpleFhiParser.py",
        "parser-fhi-aims/setup_paths.py",
        "nomad_meta_info/common.nomadmetainfo.json",
        "nomad_meta_info/meta_types.nomadmetainfo.json",
        "nomad_meta_info/fhi_aims.nomadmetainfo.json"
      ) ++ DefaultPythonInterpreter.commonFiles(),
      dirMap = Map(
        "parser-fhi-aims" -> "parsers/fhi-aims/parser/parser-fhi-aims",
        "nomad_meta_info" -> "nomad-meta-info/meta_info/nomad_meta_info",
        "python" -> "python-common/common/python/nomadcore") ++ DefaultPythonInterpreter.commonDirMapping()
)
