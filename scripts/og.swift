// Renders the landing page's social card: swift scripts/og.swift site/assets/og.png
import AppKit
import CoreText

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "site/assets/og.png"
let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
for f in ["SpaceGrotesk.woff2", "JetBrainsMono-Regular.woff2"] {
    CTFontManagerRegisterFontsForURL(root.appendingPathComponent("site/fonts/\(f)") as CFURL, .process, nil)
}
func font(_ name: String, _ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
    let base = NSFont(name: name, size: size) ?? .systemFont(ofSize: size, weight: weight)
    return NSFontManager.shared.convert(base, toHaveTrait: weight >= .semibold ? .boldFontMask : [])
}
let W = 1200, H = 630
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8, samplesPerPixel: 4,
                           hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let indigo = NSColor(red: 0.42, green: 0.36, blue: 1.0, alpha: 1), cyan = NSColor(red: 0.13, green: 0.73, blue: 0.98, alpha: 1)
let pink = NSColor(red: 0.98, green: 0.35, blue: 0.62, alpha: 1)
NSColor(red: 0.04, green: 0.04, blue: 0.10, alpha: 1).setFill()
NSRect(x: 0, y: 0, width: W, height: H).fill()
// Aurora glows.
for (c, x, y, r) in [(indigo, 980.0, 520.0, 420.0), (cyan, 1150.0, 120.0, 360.0), (pink, 700.0, -60.0, 300.0)] {
    let g = NSGradient(colors: [c.withAlphaComponent(0.55), c.withAlphaComponent(0)])!
    g.draw(fromCenter: NSPoint(x: x, y: y), radius: 0, toCenter: NSPoint(x: x, y: y), radius: r, options: [])
}
// App mark.
let mark = NSRect(x: 80, y: 430, width: 110, height: 110)
NSGradient(colors: [indigo, cyan])!.draw(in: NSBezierPath(roundedRect: mark, xRadius: 28, yRadius: 28), angle: -45)
if let s = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil)?
    .withSymbolConfiguration(.init(pointSize: 58, weight: .semibold)) {
    let tinted = NSImage(size: s.size, flipped: false) { r in s.draw(in: r); NSColor.white.set(); r.fill(using: .sourceAtop); return true }
    tinted.draw(in: NSRect(x: mark.midX - s.size.width / 2, y: mark.midY - s.size.height / 2, width: s.size.width, height: s.size.height))
}
func text(_ s: String, _ f: NSFont, _ c: NSColor, _ x: CGFloat, _ y: CGFloat) {
    (s as NSString).draw(at: NSPoint(x: x, y: y), withAttributes: [.font: f, .foregroundColor: c])
}
text("MLX Runner", font("Space Grotesk", 46, .bold), .white, 215, 462)
text("LLMs at the speed", font("Space Grotesk", 84, .bold), .white, 76, 290)
let line2 = NSMutableAttributedString(string: "of unified memory.", attributes: [.font: font("Space Grotesk", 84, .bold), .foregroundColor: NSColor.white])
line2.addAttribute(.foregroundColor, value: cyan, range: NSRange(location: 3, length: 7))
line2.addAttribute(.foregroundColor, value: NSColor(red: 0.62, green: 0.52, blue: 1, alpha: 1), range: NSRange(location: 11, length: 7))
line2.draw(at: NSPoint(x: 76, y: 190))
text("Native macOS · 100% Swift · its own Metal engine · zero dependencies", font("JetBrains Mono", 22, .regular),
     NSColor.white.withAlphaComponent(0.72), 80, 120)
text("mlx-runner.21no.de", font("JetBrains Mono", 22, .regular), cyan, 80, 58)
text("by 21no.de", font("JetBrains Mono", 22, .regular), NSColor.white.withAlphaComponent(0.55), 1000, 58)
NSGraphicsContext.current = nil
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
