package eu.nomad_lab.parsers

import org.specs2.mutable.Specification



object FhiAimsParserTests extends Specification {
  "FHIAimsParserTest" >> {
    "test with ...ZORA.out" >> {
      ParserRun.parse(FhiAimsParser,"parsers/fhi-aims/test/examples/Au2_non-periodic_scalar_ZORA.out","json-events") must_== ParseResult.ParseSuccess
    }
  }
}
