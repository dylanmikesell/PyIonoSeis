document$.subscribe(() => {
    if (window.mermaid) {
        window.mermaid.initialize({
            startOnLoad: false,
            securityLevel: "loose",
            theme: "default",
        });

        const mermaidBlocks = document.querySelectorAll(
            ".mermaid, pre > code.language-mermaid"
        );
        mermaidBlocks.forEach((block) => {
            try {
                const target =
                    block.tagName === "CODE" && block.parentElement
                        ? block.parentElement
                        : block;
                window.mermaid.run({
                    nodes: [target],
                });
            } catch (error) {
                // Keep rendering other diagrams even if one chart is invalid.
                console.error("Mermaid render failed for a diagram block.", error);
            }
        });
    }
});
