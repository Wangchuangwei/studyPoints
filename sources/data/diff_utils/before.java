public class Test {public Deriver<Map<String,String>> load(TableId key) throws Exception {return context.getServerConfFactory().getTableConfiguration(key).newDeriver(conf -> getRegexes(conf));}});LOG.info("{}", this);}}