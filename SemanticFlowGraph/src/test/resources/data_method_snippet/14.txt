public DependencyCustomizer add(String module, String classifier, String type,
			boolean transitive) {
		if (canAdd()) {
			ArtifactCoordinatesResolver artifactCoordinatesResolver = this.dependencyResolutionContext
					.getArtifactCoordinatesResolver();
			this.classNode.addAnnotation(
					createGrabAnnotation(artifactCoordinatesResolver.getGroupId(module),
							artifactCoordinatesResolver.getArtifactId(module),
							artifactCoordinatesResolver.getVersion(module), classifier,
							type, transitive));
		}
		return this;
	}