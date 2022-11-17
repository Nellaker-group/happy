"""
Coverts cell points to surrounding tissue polygon class and saves as csv
"""
import static qupath.lib.gui.scripting.QPEx.*

def classNames = ["TVilli", "SVilli", "AVilli", "MVilli", "ImIVilli", "MIVilli", "Maternal", "Chorion", "Avascular", "Fibrin", "Sprout", "Inflam"] as String[]

def slideName = getCurrentServer().getPath().split("/")[-1]
def slideNumber = slideName.split("-")[0]

def fileName = slideNumber + "_tissue_points.csv"
def saveDir = "/../projects/placenta/results/tissue_annots/" + fileName

// Get all manual annotations of tissue regions
allTissueAnnot = [:]
for (className in classNames) {
    allTissueAnnot[className] = getAnnotationObjects().findAll({ it.getPathClass().getName() == className })
}

def pointsObject = getAnnotationObjects().findAll({ it.getROI().isPoint()})
allPoints = []
for (pointClass in pointsObject) {
    allPoints.addAll(pointClass.getROI().getAllPoints())
}

// Save these to a file with columns pointx, pointy, class
def FILE_HEADER = 'px,py,class'

// Write to csv
try (PrintWriter writer = new PrintWriter(new File(saveDir))) {
    StringBuilder sb = new StringBuilder();
    sb.append(FILE_HEADER)
    sb.append('\n')

    // For each point, write the coordinate and containing region class
    def int i = 0
    for (point in allPoints) {
        def pointSaved = false
        def x = (int) point.getX()
        def y = (int) point.getY()

        for (entry in allTissueAnnot) {
            if (! pointSaved) {
                for (tissueRegion in entry.value) {
                    if (tissueRegion.getROI().contains(x, y)) {
                        sb.append(String.join(',', x.toString(), y.toString(), entry.key.toString()))
                        sb.append('\n')
                        pointSaved = true
                        break
                    }
                }
            } else {
                break
            }
        }
        if (! pointSaved) {
            sb.append(String.join(',', x.toString(), y.toString(), "Unlabelled"))
            sb.append('\n')
        }
        if (i % 100000 == 0) {
            println(i)
        }
        i++
    }

    print(sb.toString())

    writer.write(sb.toString());
    print("done!")

} catch (FileNotFoundException e) {
    print(e.getMessage())
}
