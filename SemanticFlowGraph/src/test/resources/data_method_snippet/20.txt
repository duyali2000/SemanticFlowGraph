protected Map<String, Object> generateContent() {
		Map<String, Object> content = extractContent(toPropertySource());
		postProcessContent(content);
		return content;
	}