public void addServletNames(String... servletNames) {
		Assert.notNull(servletNames, "ServletNames must not be null");
		this.servletNames.addAll(Arrays.asList(servletNames));
	}