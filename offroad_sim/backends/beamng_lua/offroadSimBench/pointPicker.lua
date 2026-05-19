-- BeamNG-side world point picker for OffroadSimBench.
--
-- Loaded as: extensions.load('offroadSimBench/pointPicker')

local M = {}
local im = ui_imgui
local logTag = "offroadSimBench_pointPicker"
local windowOpen = im and im.BoolPtr(true) or nil

local enabled = true
local sequence = 0
local lastPick = nil
local lastMarker = nil
local lastMarkerTime = 0
local lastPickTime = -1
local updateFrames = 0
local guiFrames = 0
local lastHitAvailable = false
local lastMouseClicked = false
local lastMouseDown = false
local previousMouseDown = false
local lastMouseDownEdge = false
local lastMouseReleased = false
local lastWantCaptureMouse = false
local lastMousePos = nil
local lastProcessSource = "none"
local outputFile = "settings/offroadSimBench/point_picker.json"

local function safeObjectName(object)
  if not object then return nil end
  local ok, name = pcall(function() return object:getName() end)
  if ok then return name end
  return nil
end

local function safeObjectId(object)
  if not object then return nil end
  local ok, objectId = pcall(function() return object:getID() end)
  if ok then return objectId end
  return nil
end

local function ensureOutputDirectory()
  if FS and not FS:directoryExists("settings/offroadSimBench") then
    FS:directoryCreate("settings/offroadSimBench")
  end
end

local function writePickFile(payload)
  ensureOutputDirectory()
  if jsonWriteFile then
    jsonWriteFile(outputFile, payload, true)
  end
end

local function makePick(hit)
  sequence = sequence + 1
  local pos = hit.pos
  local normal = hit.normal
  lastPick = {
    available = true,
    sequence = sequence,
    x = pos.x,
    y = pos.y,
    z = pos.z,
    distance = hit.distance,
    face = hit.face,
    object_id = safeObjectId(hit.object),
    object_name = safeObjectName(hit.object),
    normal = normal and {normal.x, normal.y, normal.z} or nil,
    source = "cameraMouseRayCast",
    timestamp = os.clock(),
  }
  lastMarker = vec3(pos)
  lastMarkerTime = os.clock()
  writePickFile(lastPick)
end

local function drawHover(hit)
  if not debugDrawer or not hit or not hit.pos then return end
  debugDrawer:drawSphere(vec3(hit.pos), 0.35, ColorF(1, 0.9, 0.05, 0.85))
end

local function drawLastMarker()
  if not debugDrawer or not lastMarker then return end
  if os.clock() - lastMarkerTime > 2.5 then return end
  debugDrawer:drawSphere(lastMarker, 0.8, ColorF(0.0, 1.0, 1.0, 0.9))
end

local function drawPickerWindow()
  if not im or not windowOpen then return end
  im.SetNextWindowSize(im.ImVec2(320, 92), im.Cond_FirstUseEver)
  if im.Begin("OffroadSimBench Picker", windowOpen) then
    im.Text("Left-click terrain to send a point to the GUI.")
    im.Text(string.format("enabled=%s sequence=%d", tostring(enabled), sequence))
    im.Text(string.format("hook=%s update=%d gui=%d", lastProcessSource, updateFrames, guiFrames))
    if lastPick then
      im.Text(string.format("last: %.2f, %.2f, %.2f", lastPick.x, lastPick.y, lastPick.z))
    end
  end
  im.End()
end

local function processMousePick(source)
  lastProcessSource = source
  if not enabled then return end
  if not cameraMouseRayCast then return end
  local hit = cameraMouseRayCast()
  lastHitAvailable = hit and hit.pos and true or false
  drawHover(hit)
  drawLastMarker()
  if not hit or not hit.pos or not im then return end
  local io = im.GetIO and im.GetIO() or nil
  lastWantCaptureMouse = io and io.WantCaptureMouse or false
  lastMouseClicked = im.IsMouseClicked(0)
  lastMouseDown = im.IsMouseDown(0)
  lastMouseDownEdge = lastMouseClicked or (lastMouseDown and not previousMouseDown)
  lastMouseReleased = im.IsMouseReleased(0)
  lastMousePos = im.GetMousePos and im.GetMousePos() or nil
  if lastMouseDownEdge and (os.clock() - lastPickTime > 0.15) then
    makePick(hit)
    lastPickTime = os.clock()
  end
  previousMouseDown = lastMouseDown
end

local function onUpdate(dt)
  updateFrames = updateFrames + 1
  processMousePick("onUpdate")
end

local function onGuiUpdate(dtReal, dtSim, dtRaw)
  guiFrames = guiFrames + 1
  drawPickerWindow()
  processMousePick("onGuiUpdate")
end

local function setEnabled(value)
  enabled = value ~= false
  return enabled
end

local function consumePickJson()
  local payload = lastPick
  lastPick = nil
  if payload then
    return jsonEncode(payload)
  end
  return jsonEncode({available = false})
end

local function consumeOrCaptureMouseJson()
  processMousePick("consume")
  return consumePickJson()
end

local function captureNowJson()
  if not cameraMouseRayCast then
    return jsonEncode({available = false, message = "cameraMouseRayCast is unavailable"})
  end
  local hit = cameraMouseRayCast()
  if not hit or not hit.pos then
    return jsonEncode({available = false, message = "cameraMouseRayCast did not hit"})
  end
  makePick(hit)
  return jsonEncode(lastPick)
end

local function statusJson()
  return jsonEncode({
    available = false,
    loaded = true,
    enabled = enabled,
    sequence = sequence,
    update_frames = updateFrames,
    gui_frames = guiFrames,
    last_hit_available = lastHitAvailable,
    last_mouse_clicked = lastMouseClicked,
    last_mouse_down = lastMouseDown,
    last_mouse_down_edge = lastMouseDownEdge,
    last_mouse_released = lastMouseReleased,
    last_want_capture_mouse = lastWantCaptureMouse,
    last_mouse_x = lastMousePos and lastMousePos.x or nil,
    last_mouse_y = lastMousePos and lastMousePos.y or nil,
    last_process_source = lastProcessSource,
  })
end

local function currentPickJson()
  if lastPick then
    return jsonEncode(lastPick)
  end
  return jsonEncode({available = false})
end

local function onExtensionLoaded()
  ensureOutputDirectory()
  writePickFile({available = false, sequence = sequence, source = "cameraMouseRayCast"})
  log("I", logTag, "OffroadSimBench point picker loaded")
  return true
end

local function onUnload()
  enabled = false
end

M.onUpdate = onUpdate
M.onGuiUpdate = onGuiUpdate
M.onExtensionLoaded = onExtensionLoaded
M.onUnload = onUnload
M.setEnabled = setEnabled
M.consumePickJson = consumePickJson
M.consumeOrCaptureMouseJson = consumeOrCaptureMouseJson
M.currentPickJson = currentPickJson
M.captureNowJson = captureNowJson
M.statusJson = statusJson

return M
