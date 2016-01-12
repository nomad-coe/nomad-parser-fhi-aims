package eu.nomad_lab.parsers
import eu.nomad_lab.DefaultPythonInterpreter
import org.{json4s => jn}

object FhiAimsParser extends SimpleExternalParserGenerator(
      name = "FhiAimsParser",
      parserInfo = jn.JObject(
        ("name" -> jn.JString("FhiAimsParser")) ::
          ("version" -> jn.JString("1.0")) :: Nil),
      mainFileTypes = Seq("text/.*"),
      mainFileRe = """          Invoking FHI-aims ...
          Version """.r,
      cmd = Seq(DefaultPythonInterpreter.python2Exe(), "${envDir}/SimpleFhiParser.py",
        "--uri", "${mainFileUri}", "${mainFilePath}"),
      resList = Seq(
        "parser-fhi-aims/SimpleFhiParser.py",
        "parser-fhi-aims/setup_paths.py"
      ) ++ DefaultPythonInterpreter.commonFiles(),
      dirMap = Map(
        "parser-fhi-aims" -> "parsers/fhi-aims/parser/parser-fhi-aims",
        "nomadcore" -> "python-common/python/nomadcore") ++ DefaultPythonInterpreter.commonDirMapping()
)
