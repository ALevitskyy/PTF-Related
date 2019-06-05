from RandomCageOverlay import default_cage_overlay
from CageMaker import load_cv2_RGB
import cv2
from RandomInfoOverlay import default_info_overlay
import pickle
def testRandomCageOverlay():
    test_image = load_cv2_RGB("test_image.png")
    test_mask = load_cv2_RGB("test_mask.png")
    test_result = default_cage_overlay(image = test_image, mask = test_mask)
    cv2.imwrite("testresultimage.png",cv2.cvtColor(test_result["image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite("testresultmask.png",cv2.cvtColor(test_result["mask"], cv2.COLOR_RGB2BGR))
    #pickle.dump( default_cage_overlay,
    #open( "overlays/default_cage_overlay.pkl", "wb" ) )
#testRandomCageOverlay()
def testRandomInfoOverlay():
    test_image = load_cv2_RGB("test_image.png")
    test_mask = load_cv2_RGB("test_mask.png")
    test_result = default_info_overlay(image = test_image, mask = test_mask)
    cv2.imwrite("testoverlayimage.png",cv2.cvtColor(test_result["image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite("testoverlaymask.png",cv2.cvtColor(test_result["mask"], cv2.COLOR_RGB2BGR))
    #pickle.dump( default_info_overlay,
    #open( "overlays/default_info_overlay.pkl", "wb" ) )
testRandomInfoOverlay()

