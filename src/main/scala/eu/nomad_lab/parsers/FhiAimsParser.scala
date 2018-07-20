/*
 * Copyright 2015-2018 Franz Knuth, Fawzi Mohamed, Wael Chibani, Ankit Kariryaa, Lauri Himanen, Danio Brambila
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

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
        ("nomadCoreVersion" -> jn.JObject(lab.NomadCoreVersionInfo.toMap.map {
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
  cmd = Seq(DefaultPythonInterpreter.pythonExe(), "${envDir}/parsers/fhi-aims/parser/parser-fhi-aims/fhiaimsparser/FhiAimsParser.py",
    "--uri", "${mainFileUri}", "${mainFilePath}"),
  resList = Seq(
    "parser-fhi-aims/fhiaimsparser/__init__.py",
    "parser-fhi-aims/fhiaimsparser/parser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsCommon.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsControlInParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsBandParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsDosParser.py",
    "parser-fhi-aims/fhiaimsparser/setup_paths.py",
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

object FhiAimsControlInParser extends SimpleExternalParserGenerator(
  name = "FhiAimsControlInParser",
  parserInfo = jn.JObject(
    ("name" -> jn.JString("FhiAimsControlInParser")) ::
      ("parserId" -> jn.JString("FhiAimsControlInParser" + lab.FhiAimsVersionInfo.version)) ::
      ("versionInfo" -> jn.JObject(
        ("nomadCoreVersion" -> jn.JObject(lab.NomadCoreVersionInfo.toMap.map {
          case (k, v) => k -> jn.JString(v.toString)
        }(breakOut): List[(String, jn.JString)])) ::
          (lab.FhiAimsVersionInfo.toMap.map {
            case (key, value) =>
              (key -> jn.JString(value.toString))
          }(breakOut): List[(String, jn.JString)])
      )) :: Nil
  ),
  mainFileTypes = Seq("text/.*"),
  mainFileRe = """\s*species\s+[A-Z][a-z]*\s*\n?.*\s*\n?.*\s*\n?.*\s*\n?.*\s*\n?\s*nucleus\s+[0-9]+""".r,
  cmd = Seq(DefaultPythonInterpreter.pythonExe(), "${envDir}/parsers/fhi-aims/parser/parser-fhi-aims/fhiaimsparser/FhiAimsControlInParser.py",
    "--uri", "${mainFileUri}", "${mainFilePath}"),
  resList = Seq(
    "parser-fhi-aims/fhiaimsparser/__init__.py",
    "parser-fhi-aims/fhiaimsparser/parser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsCommon.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsControlInParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsBandParser.py",
    "parser-fhi-aims/fhiaimsparser/FhiAimsDosParser.py",
    "parser-fhi-aims/fhiaimsparser/setup_paths.py",
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
