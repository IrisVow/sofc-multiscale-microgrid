import java.nio.file.*;
public final class ProjectPaths {
    public static String resolve(String relative) {
        String override = System.getenv("SOFC_PROJECT_ROOT");
        Path root = override == null ? Paths.get("").toAbsolutePath() : Paths.get(override).toAbsolutePath();
        while (root != null && !Files.isRegularFile(root.resolve(".sofc-project"))) root = root.getParent();
        if (root == null) throw new IllegalStateException("Run from the SOFC repository or set SOFC_PROJECT_ROOT");
        String resolved = root.resolve(relative).normalize().toString().replace('\\', '/');
        return relative.endsWith("/") ? resolved + "/" : resolved;
    }
}
