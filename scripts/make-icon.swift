// Renders the MLX Runner app icon (1024×1024 PNG) with AppKit.
// Usage: swift scripts/make-icon.swift <output.png>
import AppKit

let size: CGFloat = 1024
let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon.png"
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size), pixelsHigh: Int(size), bitsPerSample: 8,
                           samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let ctx = NSGraphicsContext.current!.cgContext

// macOS squircle-ish rounded rect with margin
let inset: CGFloat = 100
let rect = CGRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
let path = CGPath(roundedRect: rect, cornerWidth: 185, cornerHeight: 185, transform: nil)

// drop shadow
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -18), blur: 40, color: NSColor.black.withAlphaComponent(0.35).cgColor)
ctx.addPath(path); ctx.setFillColor(NSColor.black.cgColor); ctx.fillPath()
ctx.restoreGState()

// background gradient (indigo → cyan)
ctx.saveGState()
ctx.addPath(path); ctx.clip()
let bg = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [
    NSColor(red: 0.20, green: 0.10, blue: 0.55, alpha: 1).cgColor,
    NSColor(red: 0.42, green: 0.36, blue: 1.00, alpha: 1).cgColor,
    NSColor(red: 0.13, green: 0.73, blue: 0.98, alpha: 1).cgColor,
] as CFArray, locations: [0, 0.55, 1])!
ctx.drawLinearGradient(bg, start: CGPoint(x: rect.minX, y: rect.maxY), end: CGPoint(x: rect.maxX, y: rect.minY), options: [])
// soft glow
let glow = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [
    NSColor(white: 1, alpha: 0.35).cgColor, NSColor(white: 1, alpha: 0).cgColor] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(glow, startCenter: CGPoint(x: size * 0.35, y: size * 0.72), startRadius: 0,
                       endCenter: CGPoint(x: size * 0.35, y: size * 0.72), endRadius: 520, options: [])
ctx.restoreGState()

// Symbol: sparkles over a chip outline
func symbol(_ name: String, _ pt: CGFloat, _ weight: NSFont.Weight) -> NSImage? {
    let cfg = NSImage.SymbolConfiguration(pointSize: pt, weight: weight)
        .applying(NSImage.SymbolConfiguration(paletteColors: [.white]))
    return NSImage(systemSymbolName: name, accessibilityDescription: nil)?.withSymbolConfiguration(cfg)
}
if let chip = symbol("cpu", 470, .ultraLight) {
    let s = chip.size
    chip.draw(in: CGRect(x: (size - s.width) / 2, y: (size - s.height) / 2, width: s.width, height: s.height),
              from: .zero, operation: .sourceOver, fraction: 0.28)
}
if let sparkle = symbol("sparkles", 330, .semibold) {
    let s = sparkle.size
    ctx.setShadow(offset: CGSize(width: 0, height: -6), blur: 24, color: NSColor(red: 0.1, green: 0, blue: 0.4, alpha: 0.5).cgColor)
    sparkle.draw(in: CGRect(x: (size - s.width) / 2, y: (size - s.height) / 2, width: s.width, height: s.height))
}
NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
print("wrote \(out)")
