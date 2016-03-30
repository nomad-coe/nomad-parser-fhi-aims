package eu.nomad_lab.parsers

import org.specs2.mutable.Specification

object FhiAimsParserTests extends Specification {
  "FHIAimsParserTest" >> {
    "test with json-events" >> {
      ParserRun.parse(FhiAimsParser, "parsers/fhi-aims/test/examples/Au2_non-periodic_geometry_optimization.out", "json-events") must_== ParseResult.ParseSuccess
    }
    "test with json" >> {
      ParserRun.parse(FhiAimsParser, "parsers/fhi-aims/test/examples/Au2_non-periodic_geometry_optimization.out", "json") must_== ParseResult.ParseSuccess
    }
  }
}
